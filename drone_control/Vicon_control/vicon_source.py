"""Live world-frame pose from Vicon, for closed-loop control.

Owns the ONE `UdpRigidBodiesViCON` receiver (UdpReceiver_datacollection.py binds
:51001 without SO_REUSEADDR, so only one socket can exist). The control loop calls
get_pose() each tick; the recorder is handed this same receiver (via
ViconRecorder.prepare(external_udp=...)) so control + recording share one stream.

World frame (verified against recordings/20260609_135349):
  - Z is UP (altitude). X, Y are horizontal.
  - yaw is rotation about world Z, from the Vicon quaternion.
Velocities are world-frame, from the lab `Differentiator` (diff_steps=2), stepped
on the packet timestamps so they're correct regardless of control-loop jitter.

This is read-only feedback — it never touches the serial link or arming.
"""
import os
import sys
import math
import time
import socket

# The Vicon parser + receiver live in the lab's pycode_ViCON/ (the base template
# for all Vicon IO in this repo). Same path the recorders use.
_PYCODE_VICON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pycode_ViCON")
if _PYCODE_VICON not in sys.path:
    sys.path.insert(0, _PYCODE_VICON)
from UdpReceiver_datacollection import (  # noqa: E402
    UdpRigidBodiesViCON, DataProcessorViCON, Differentiator,
)

VICON_UDP_IP = "0.0.0.0"
VICON_UDP_PORT = 51001
_PROBE_BLOCK = 1024
_PROBE_TIMEOUT_S = 5.0


def quat_to_yaw(qx, qy, qz, qw):
    """Yaw (rad) about world Z from a quaternion. Z-up convention.
    SIGN to be confirmed empirically in DRY_RUN (hand-yaw the drone)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def drone_roll_pitch(qx, qy, qz, qw):
    """Drone-body roll & pitch (rad) from the Vicon quaternion, in the CONTROLLER's
    sign convention (roll>0 = rolled RIGHT, pitch>0 = nose DOWN) so the ACRO
    angle→rate loop can difference them directly against desired_roll/desired_pitch.

    EMPIRICALLY VERIFIED 2026-06-16 (acro_attitude_check.py, 3-pose bench test): the
    Vicon rigid body is mounted ~90° rotated from the drone's roll/pitch axes (same
    root cause as VICON_YAW_OFFSET_DEG=90), so the STANDARD aerospace roll/pitch
    extracted from the quaternion come out SWAPPED relative to the drone:
        standard roll  (about quat X)  →  drone PITCH  (nose-down measured NEGATIVE)
        standard pitch (about quat Y)  →  drone ROLL   (right-down measured POSITIVE)
    The extraction is also HEADING-INVARIANT (nose-down read the same facing +Y and
    facing +X after a 90° turn), so NO yaw rotation is applied here. Bench data:
    pose1 (face+Y, nose-down) std[R-37 P~0]; pose2 (roll-right) std[R~-1.5 P+35];
    pose3 (face+X, nose-down) std[R-38 P~0] == pose1 → heading-invariance confirmed."""
    # standard aerospace ZYX extraction
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    std_roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    std_pitch = math.asin(sinp)
    # 90° mount swap + sign into the controller convention (see docstring / bench data)
    drone_pitch = -std_roll     # nose-down POSITIVE (matches desired_pitch>0)
    drone_roll = std_pitch      # roll-right POSITIVE (matches desired_roll>0)
    return drone_roll, drone_pitch


class ViconPoseSource:
    """Single Vicon receiver + per-tick pose read for the controller.

    Lifecycle:
      prepare()  -> probe + start the receiver thread (fail-soft; never hangs).
      get_pose() -> latest world-frame pose+velocity (called each control tick).
      .udp       -> the shared receiver, to inject into ViconRecorder.
    """

    def __init__(self, port=VICON_UDP_PORT, ip=VICON_UDP_IP, body_index=1,
                 yaw_offset_deg=0.0):
        self.port = port
        self.ip = ip
        self.body_index = body_index      # b1 = the drone
        # Constant added to the Vicon quaternion-yaw so the reported yaw is the
        # drone's TRUE heading (Vicon's rigid-body frame is rotated from the drone's
        # roll/pitch axes — see config.VICON_YAW_OFFSET_DEG). Applied here so every
        # consumer (target capture, body decomposition, yaw hold) is consistent.
        self.yaw_offset = math.radians(yaw_offset_deg)
        self.udp = None
        self.dp = None
        self.sample_rate = None
        self.num_bodies = None
        self.status = "idle"
        self._diff_x = Differentiator(diff_steps=2)
        self._diff_y = Differentiator(diff_steps=2)
        self._diff_z = Differentiator(diff_steps=2)
        self._last_udp_time = None        # packet wall-time of the last read

    def _stream_present(self):
        """Fail-soft probe: is anything streaming on the Vicon port right now?
        Avoids the lab receiver's un-timeouted blocking get_sample_rate when Vicon
        is off, so the controller never hangs at startup."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.ip, self.port))
            s.settimeout(_PROBE_TIMEOUT_S)
            s.recvfrom(_PROBE_BLOCK)
            return True
        except socket.timeout:
            return False
        finally:
            s.close()

    def prepare(self):
        """Probe, then start the receiver + parser. Returns True if Vicon is live
        and streaming, False otherwise. Never raises, never hangs."""
        if self.udp is not None:
            return True
        if not self._stream_present():
            self.status = f"no Vicon stream on :{self.port}"
            return False
        try:
            self.udp = UdpRigidBodiesViCON(udp_ip=self.ip, udp_port=self.port)
            self.udp.start_thread()
            self.sample_rate = self.udp.sample_rate
            self.num_bodies = self.udp.num_bodies
            self.dp = DataProcessorViCON(self.num_bodies, self.sample_rate)
            self.status = (f"ready ({self.num_bodies} bodies, "
                           f"{self.sample_rate:.0f} Hz)")
            return True
        except Exception as e:  # noqa: BLE001 — never let Vicon setup reach flight
            self.status = f"vicon prepare error: {e}"
            self.udp = None
            return False

    def get_pose(self, now=None):
        """Latest world-frame pose for the body. Returns a dict or None if no
        packet has arrived yet. Velocities are stepped on the packet timestamp.

        keys: x, y, z, yaw (rad), roll, pitch (rad — drone body, controller
        convention: roll>0 right / pitch>0 nose-down), vx, vy, vz, t_packet (wall),
        age_s (since packet)
        """
        if self.udp is None:
            return None
        now = time.time() if now is None else now
        data_raw, udp_time = self.udp.get_data()
        if data_raw is None:
            return None
        data, _ = self.dp.process_data(data_raw)
        b = data.get(self.body_index)
        if b is None:
            return None
        x, y, z = b["x"], b["y"], b["z"]
        yaw = quat_to_yaw(b["qx"], b["qy"], b["qz"], b["qw"]) + self.yaw_offset
        # Body roll/pitch for the ACRO leveling loop (no yaw offset — the extraction
        # is heading-invariant; see drone_roll_pitch). Harmless in Angle mode.
        roll, pitch = drone_roll_pitch(b["qx"], b["qy"], b["qz"], b["qw"])
        # Step velocities only on a genuinely new packet so a stalled stream
        # doesn't inject dt-driven noise (Differentiator no-ops when dt==0).
        if udp_time != self._last_udp_time:
            self._diff_x.step(x, udp_time)
            self._diff_y.step(y, udp_time)
            self._diff_z.step(z, udp_time)
            self._last_udp_time = udp_time
        return {
            "x": x, "y": y, "z": z, "yaw": yaw,
            "roll": roll, "pitch": pitch,
            "vx": self._diff_x.data_rate,
            "vy": self._diff_y.data_rate,
            "vz": self._diff_z.data_rate,
            "t_packet": udp_time,
            "age_s": now - udp_time,
        }

    def stop(self):
        """Stop the receiver thread (it's non-daemon, so call this on exit). The
        worker unblocks on the next packet — fine while Vicon is still streaming."""
        if self.udp is not None:
            try:
                self.udp.stop_thread()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                pass
