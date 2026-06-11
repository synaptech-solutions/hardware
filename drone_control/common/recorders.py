"""Per-session recorders + session.json writer — the synced data-collection pipeline.

Extracted verbatim from data_logging/joystick_flight.py so the manual data logger
AND the autonomous Vicon controller record EVERY flight the same way, all stamped
on one clock (t_rel = wall - t0) so they merge deterministically via
combine_flight.py:
  - VideoRecorder    drone feed → video.mkv (lossless FFV1; + video_frames.csv real capture times)
  - ViconRecorder    Vicon pose @100 Hz → vicon.mat
  - CommandLogger    every outgoing RC frame → commands.csv (16 ch + full joystick)
  - TelemetryLogger  incoming CRSF telemetry → telemetry.csv (+ telemetry_raw.csv)
  - _write_session_json   t0 / per-stream offsets / config snapshot → session.json

cv2/numpy/scipy are optional: a missing dep disables that recorder and flight
continues (run with the repo .venv to get them).

The ONE change vs the original: ViconRecorder.prepare() can take an EXTERNAL,
already-running receiver+processor (external_udp/external_dp). UdpRigidBodiesViCON
binds :51001 without SO_REUSEADDR, so only one socket can exist — the Vicon
controller's vicon_source owns it and injects it here so control + recording share
one stream. With no external receiver it self-owns one exactly as before.
"""
import os
import sys
import csv
import json
import time
import shutil
import datetime
import threading
import subprocess

from . import channels

# OpenCV is only needed for video recording. Import it optionally so flight still
# works (recording disabled) under a Python without cv2 — e.g. system python3 vs
# the repo's .venv (python3.12) which has cv2.
try:
    import cv2  # noqa: E402
    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False

# We encode with an external ffmpeg subprocess (FFV1/MKV — see VideoRecorder), so
# its presence gates video recording just like cv2 does. shutil.which resolves it
# from PATH once at import; None → video recording disables itself, flight runs on.
FFMPEG_BIN = shutil.which("ffmpeg")
FFMPEG_OK = FFMPEG_BIN is not None

# Vicon pose recorder deps. Optional like cv2 — if numpy/scipy/the lab parser are
# missing, vicon recording disables itself (VICON_OK=False) and flight continues.
# The parser is the lab's, in the sibling pycode_ViCON/ folder.
_PYCODE_VICON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pycode_ViCON")
if _PYCODE_VICON not in sys.path:
    sys.path.insert(0, _PYCODE_VICON)
try:
    import socket  # noqa: E402
    import numpy as np  # noqa: E402
    import scipy.io as sio  # noqa: E402
    # Same primitives UdpReceiver_datacollection.py uses — it is the base template
    # for ALL Vicon logging in this repo, so this path stays identical to it.
    from UdpReceiver_datacollection import (  # noqa: E402
        DataProcessorViCON, UdpRigidBodiesViCON, RealTimeSleeper, Differentiator,
    )
    VICON_OK = True
    _VICON_ERR = None
except Exception as _e:  # noqa: BLE001 — missing dep just disables vicon
    VICON_OK = False
    _VICON_ERR = _e

VICON_UDP_IP = "0.0.0.0"
VICON_UDP_PORT = 51001
_VICON_BLOCK = 1024
_VICON_PROBE_TIMEOUT_S = 5.0   # no Vicon traffic within this at prepare() → disable
_VICON_SAMPLE_DT = 0.01        # 100 Hz logging loop, same as the UdpReceiver template

CSI = "\033["


class VideoRecorder:
    """Records the drone's video feed to a LOSSLESS file in a background thread.

    Frames are captured with cv2.VideoCapture (camera-native MJPG) and piped raw
    to an ffmpeg subprocess that encodes FFV1 inside an MKV — mathematically
    lossless (bit-exact vs the captured BGR frame), intra-only (-g 1: every frame
    an independent keyframe), with per-frame CRCs (-slicecrc 1) so silent storage
    corruption is detectable years later. We pipe to ffmpeg rather than use
    cv2.VideoWriter because OpenCV can't emit FFV1/MKV and writes a fake
    constant-rate clock (opencv #23403) — here the container PTS is irrelevant on
    read anyway: recover any frame BY INTEGER INDEX, then look up its true capture
    time in video_frames.csv (file frame N <-> csv row N, written in lockstep).

    Camera open + per-frame read/encode happen off the main loop so they can
    never stall the 50 Hz RC stream. start()/stop() are called from the main
    loop on switch edges; start() returns immediately (the thread opens the
    camera, ~0.5-1 s, so the first second of footage may be missed)."""

    def __init__(self, device_index, width, height):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._thread = None
        self._stop = threading.Event()
        self.recording = False
        self.status = ("idle" if (CV2_OK and FFMPEG_OK)
                       else "disabled (no cv2)" if not CV2_OK
                       else "disabled (no ffmpeg)")
        self.path = None
        self.frames = 0
        self.fps = None
        # Frame size ffmpeg was launched with — set from the FIRST captured frame
        # (the camera may ignore the requested W/H). Any later frame of a different
        # size is dropped (not written, not logged) to keep file<->csv aligned.
        self.enc_w = None
        self.enc_h = None
        self.t0 = None
        # Per-frame capture wall-clock times (frame_idx, t_wall), written to
        # video_frames.csv on stop. Real capture times beat assuming a constant
        # fps: they expose drops/jitter and let the merge align the video exactly.
        self.frame_times = []
        # Wall-clock time the first frame was actually captured. The camera takes
        # ~0.5-1 s to open, so frame 0 lags the session t0 by this much — the
        # combine uses (first_frame_wall - t0) to align video to the data.
        self.first_frame_wall = None
        # Latest captured frame, shared (read-only) with the main loop so it can
        # show a live preview window. Guarded by a lock; capture stays in _run.
        self._frame_lock = threading.Lock()
        self._latest_frame = None

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def start(self, t0, out_path):
        if self.recording or not (CV2_OK and FFMPEG_OK):
            return
        # Make sure any prior recording's thread has fully finalized its file.
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.t0 = t0
        self.path = out_path
        self.first_frame_wall = None
        self.frame_times = []
        self.recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.recording:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.recording = False

    def _spawn_ffmpeg(self, w, h):
        """Start the FFV1/MKV encoder. Raw BGR24 frames are piped to its stdin;
        '-pix_fmt gbrp' on the output keeps it bit-exact (planar RGB — no YUV
        chroma subsampling or color-conversion rounding), '-g 1' makes every
        frame an independent keyframe, '-slicecrc 1' stamps a CRC on each so
        corruption is detectable. The '-framerate' only sets the container's
        nominal playback rate; it is NOT the real timing (see video_frames.csv).
        Returns the Popen, or None if launch failed."""
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{w}x{h}", "-framerate", f"{self.fps:g}", "-i", "-",
            "-an", "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
            "-g", "1", "-slicecrc", "1", "-pix_fmt", "gbrp", self.path,
        ]
        try:
            return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception:
            return None

    def _run(self):
        cap = proc = None
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            name = os.path.basename(self.path)
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.status = f"camera /dev/video{self.device_index} open FAILED"
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rep = cap.get(cv2.CAP_PROP_FPS)
            self.fps = rep if rep and rep > 1 else 30.0
            self.frames = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                cap_t = time.time()
                # Launch the encoder on the first real frame, sized to what the
                # camera ACTUALLY returns (it may ignore the requested W/H).
                if proc is None:
                    self.enc_h, self.enc_w = frame.shape[:2]
                    proc = self._spawn_ffmpeg(self.enc_w, self.enc_h)
                    if proc is None:
                        self.status = "ffmpeg launch FAILED"
                        return
                    self.status = (f"REC {name} (FFV1 {self.enc_w}x{self.enc_h} "
                                   f"@ {self.fps:.0f}fps)")
                if self.first_frame_wall is None:
                    self.first_frame_wall = cap_t
                # Drop any odd-sized frame WITHOUT logging it, so file frame N
                # stays exactly aligned with video_frames.csv row N.
                if (frame.shape[1], frame.shape[0]) != (self.enc_w, self.enc_h):
                    continue
                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, ValueError):
                    self.status = "ffmpeg pipe closed (encoder died)"
                    break
                self.frame_times.append((self.frames, cap_t))
                self.frames += 1
                with self._frame_lock:
                    self._latest_frame = frame
            self.status = f"saved {name} ({self.frames} frames)"
        except Exception as e:  # never let a recorder fault take down flight
            self.status = f"recorder error: {e}"
        finally:
            # Close the pipe so ffmpeg flushes and finalizes the MKV, then reap it.
            if proc is not None:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=10.0)
                except Exception:
                    proc.kill()
            if cap is not None:
                cap.release()
            self._save_frame_times()
            self.recording = False

    def _save_frame_times(self):
        """Write video_frames.csv (frame_idx, t_rel, t_wall) alongside the mp4."""
        if not self.frame_times or not self.path:
            return
        t0 = self.t0 or self.first_frame_wall or 0.0
        out = os.path.join(os.path.dirname(self.path) or ".", "video_frames.csv")
        try:
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "t_rel", "t_wall"])
                for idx, tw in self.frame_times:
                    w.writerow([idx, tw - t0, tw])
        except Exception:  # never let a logging fault take down flight
            pass


class ViconRecorder:
    """Records Vicon pose to a per-session .mat, built on the SAME primitives as
    UdpReceiver_datacollection.py: `UdpRigidBodiesViCON` (threaded receiver + the
    original startup sample-rate determination), `DataProcessorViCON` (parser), a
    100 Hz `RealTimeSleeper` loop, and `Differentiator` for b1 velocities. Output
    matches that template — `exptime`, `Abs_time` (from the shared trigger t0),
    `b1_x`..`b1_qw` (+ any extra bodies), `b1_x_dot`/`b1_y_dot`/`b1_z_dot`.

    prepare() does the connect + sample-rate measurement ONCE at startup behind a
    fail-soft probe so flight still works if Vicon isn't streaming, then the
    receiver runs continuously; start()/stop() just gate recording so each session
    begins logging instantly at t0 (no per-flick measurement delay) — keeping the
    deterministic sync with video + blackbox intact.

    SHARED RECEIVER: pass external_udp + external_dp to prepare() to log from an
    already-running receiver (the Vicon controller owns one for control). With
    none, it self-owns one exactly as the data logger does."""

    def __init__(self, port=VICON_UDP_PORT, ip=VICON_UDP_IP):
        self.port = port
        self.ip = ip
        self.udp = None          # UdpRigidBodiesViCON — built/injected in prepare()
        self.dp = None           # DataProcessorViCON
        self._owns_udp = False   # True only if WE created the receiver
        self.sample_rate = None  # measured by the startup determination
        self.num_bodies = None
        self._thread = None
        self._stop = threading.Event()
        self.recording = False
        self.status = "idle" if VICON_OK else f"disabled ({_VICON_ERR})"
        self.path = None
        self.samples = 0
        self.first_packet_wall = None
        self.t0_wall = None

    def _stream_present(self):
        """Fail-soft probe: is anything streaming on the Vicon port right now?
        Lets us skip the lab receiver's un-timeouted blocking get_sample_rate
        when Vicon is off, so flight never hangs at startup."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.ip, self.port))
            s.settimeout(_VICON_PROBE_TIMEOUT_S)
            s.recvfrom(_VICON_BLOCK)
            return True
        except socket.timeout:
            return False
        finally:
            s.close()

    def prepare(self, external_udp=None, external_dp=None):
        """Connect + run the original startup sample-rate determination ONCE (or
        adopt an injected receiver). Returns True if Vicon is live and ready to
        record, False (disabled) if not — never raises, never hangs flight. Safe
        to call again; no-op once prepared."""
        if not VICON_OK or self.udp is not None:
            return self.udp is not None
        # Inject an already-running receiver (shared with a controller's pose
        # source). We take our OWN DataProcessorViCON — process_data() mutates its
        # data_list, so control + recording must not share one parser.
        if external_udp is not None:
            self.udp = external_udp
            self._owns_udp = False
            self.sample_rate = getattr(external_udp, "sample_rate", None)
            self.num_bodies = getattr(external_udp, "num_bodies", None)
            self.dp = external_dp or DataProcessorViCON(self.num_bodies,
                                                        self.sample_rate)
            self.status = (f"ready (shared receiver, {self.num_bodies} bodies, "
                           f"{self.sample_rate:.0f} Hz)")
            return True
        if not self._stream_present():
            self.status = f"no Vicon stream on :{self.port} — disabled"
            return False
        try:
            # UdpRigidBodiesViCON.__init__ runs get_sample_rate() (times ~1000
            # packets); start_thread() then reads num_bodies from the header.
            self.udp = UdpRigidBodiesViCON(udp_ip=self.ip, udp_port=self.port)
            self.udp.start_thread()
            self._owns_udp = True
            self.sample_rate = self.udp.sample_rate
            self.num_bodies = self.udp.num_bodies
            self.dp = DataProcessorViCON(self.num_bodies, self.sample_rate)
            self.status = (f"ready ({self.num_bodies} bodies, "
                           f"{self.sample_rate:.0f} Hz)")
            return True
        except Exception as e:  # noqa: BLE001 — never let vicon setup reach flight
            self.status = f"vicon prepare error: {e}"
            self.udp = None
            return False

    def start(self, t0_wall, out_path):
        if self.recording or self.udp is None:   # prepare() must have succeeded
            return
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.t0_wall = t0_wall
        self.path = out_path
        self.samples = 0
        self.first_packet_wall = None
        self.recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.recording:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.recording = False

    def _run(self):
        # Same loop body as the UdpReceiver template: 100 Hz RealTimeSleeper,
        # get_data() (latest packet from the receiver thread), process, then
        # Differentiate b1 x/y/z for velocities. Records from the first tick, so
        # the session starts at t0 (receiver is already running from prepare()).
        names = list(self.dp.save_list_name) + ["b1_x_dot", "b1_y_dot", "b1_z_dot"]
        rts = RealTimeSleeper(_VICON_SAMPLE_DT)
        diff_x = Differentiator(diff_steps=2)
        diff_y = Differentiator(diff_steps=2)
        diff_z = Differentiator(diff_steps=2)
        abs_time = []
        rows = []
        try:
            rts.init()
            while not self._stop.is_set():
                data_raw, udp_time = self.udp.get_data()
                data, save_list_data = self.dp.process_data(data_raw)
                if self.first_packet_wall is None:
                    self.first_packet_wall = time.time()
                    self.status = f"REC vicon ({self.num_bodies} bodies)"
                diff_x.step(data[1]["x"], udp_time)
                diff_y.step(data[1]["y"], udp_time)
                diff_z.step(data[1]["z"], udp_time)
                abs_time.append(time.time() - self.t0_wall)
                rows.append(list(save_list_data)
                            + [diff_x.data_rate, diff_y.data_rate, diff_z.data_rate])
                self.samples = len(rows)
                rts.sleep()
            self._save(abs_time, rows, names)
        except Exception as e:  # noqa: BLE001 — never let vicon faults reach flight
            self.status = f"vicon recorder error: {e}"
        finally:
            self.recording = False

    def _save(self, abs_time, rows, names):
        if not rows:
            self.status = "no Vicon samples recorded"
            return
        arr = np.asarray(rows, dtype=float)
        out = {
            "exptime": (datetime.datetime.fromtimestamp(self.t0_wall)
                        .strftime("%Y%m%d_%H%M%S")),
            "Abs_time": np.asarray(abs_time, dtype=float),
            "t0_wall": float(self.t0_wall),
            "first_packet_wall": float(self.first_packet_wall or self.t0_wall),
            "num_samples": len(rows),
            "sample_rate_hz": float(self.sample_rate or 0.0),
        }
        for i, nm in enumerate(names):
            out[nm] = arr[:, i]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        sio.savemat(self.path, out)
        self.status = f"saved {os.path.basename(self.path)} ({len(rows)} samples)"


class CommandLogger:
    """Buffers every outgoing RC frame in RAM during a session, writes
    commands.csv on stop. Captures ALL 16 CRSF channels (not just the mapped
    gimbals) plus the FULL joystick snapshot (every axis + button), so nothing
    we sent — or that the operator touched — is ever lost. No file I/O on the
    50 Hz hot path: accumulate now, save at end.

    Each row is stamped t_rel = t_wall - t0, the SAME clock as Vicon Abs_time, so
    the merge is a direct join. t0 is the session trigger (begin_session)."""

    def __init__(self):
        self.rows = []
        self.n_axes = 0
        self.n_buttons = 0
        self.t0 = None
        self.path = None

    def start(self, t0, path, n_axes, n_buttons):
        self.rows = []
        self.t0 = t0
        self.path = path
        self.n_axes = n_axes
        self.n_buttons = n_buttons

    def log(self, t_wall, ch, armed, record_on, axes, buttons):
        """One row: timestamps, all 16 channel µs, arm/record flags, then the
        raw joystick axes and buttons (blank where the device didn't report)."""
        row = [t_wall - self.t0, t_wall]
        row.extend(int(c) for c in ch)
        row.append(1 if armed else 0)
        row.append(1 if record_on else 0)
        row.extend(axes.get(i, "") for i in range(self.n_axes))
        row.extend(buttons.get(i, "") for i in range(self.n_buttons))
        self.rows.append(row)

    def save(self):
        if not self.rows or self.path is None:
            return
        header = (["t_rel", "t_wall"]
                  + [f"ch{i:02d}_us" for i in range(16)]
                  + ["armed", "record_on"]
                  + [f"ax{i}" for i in range(self.n_axes)]
                  + [f"btn{i}" for i in range(self.n_buttons)])
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(self.rows)


# Typed-telemetry column order. Each decoded frame fills only its own columns;
# the rest stay blank (one row per incoming frame, tagged by `type`).
TELEM_COLS = [
    "t_rel", "t_wall", "type",
    "att_pitch_deg", "att_roll_deg", "att_yaw_deg",
    "bat_v", "bat_a", "bat_mah", "bat_pct",
    "up_lq", "dn_lq", "up_rssi_dbm", "dn_rssi_dbm",
    "up_snr_db", "dn_snr_db", "rf_mode", "active_ant", "up_tx_pwr_idx",
    "imu_ax_g", "imu_ay_g", "imu_az_g",
    "imu_gx_dps", "imu_gy_dps", "imu_gz_dps", "imu_mag",
    "flight_mode", "device_addr", "device_name",
]


class TelemetryLogger:
    """Buffers incoming drone telemetry in RAM during a session, writes two CSVs
    on stop (no hot-path I/O):
      - telemetry.csv:     typed decode of the known CRSF frames (attitude,
                           battery, link stats, flight mode, device info), one
                           row per frame, stamped at laptop receive time.
      - telemetry_raw.csv: EVERY incoming frame as (type, len, payload_hex) — the
                           lossless safety net, re-decodable offline.

    Both stamped t_rel = t_wall - t0 (same clock as Vicon Abs_time / commands)."""

    def __init__(self):
        self.typed = []
        self.raw = []
        self.t0 = None
        self.path = None
        self.raw_path = None

    def start(self, t0, path, raw_path):
        self.typed = []
        self.raw = []
        self.t0 = t0
        self.path = path
        self.raw_path = raw_path

    def log_typed(self, t_wall, type_str, fields):
        row = {"t_rel": t_wall - self.t0, "t_wall": t_wall, "type": type_str}
        row.update(fields)
        self.typed.append(row)

    def log_raw(self, t_wall, ftype, payload):
        self.raw.append([t_wall - self.t0, t_wall,
                         f"0x{ftype:02X}", len(payload), payload.hex()])

    def save(self):
        if self.path and self.typed:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TELEM_COLS, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.typed)
        if self.raw_path and self.raw:
            os.makedirs(os.path.dirname(self.raw_path) or ".", exist_ok=True)
            with open(self.raw_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_rel", "t_wall", "frame_type", "length", "payload_hex"])
                w.writerows(self.raw)


def write_session_json(session, recorder, vicon, cmd_log, telem, cal, port, baud,
                       vicon_off_reason=None, extra=None):
    """Write per-session metadata so combine_flight.py + analysis can align all
    streams to the shared trigger t0, and so the dataset is self-describing — the
    config/calibration snapshot records what each channel/axis meant at capture.

    `extra` (dict) is merged in so a controller can record its own provenance
    (e.g. target setpoint, gains) without changing the shared schema."""
    meta = {
        "stamp": session["stamp"],
        "t0_wall": session["t0"],
        "t0_human": datetime.datetime.fromtimestamp(session["t0"]).isoformat(),
        "aux2_on_us": channels.AUX2_HIGH_US,
        "note": ("All laptop streams begin at the switch flick (t0); commands.csv, "
                 "telemetry.csv and vicon Abs_time are stamped t_rel = t_wall - t0. "
                 "Blackbox time zero-bases to its first sample (≈ same flick); video "
                 "lags t0 by video.start_offset_s (use video_frames.csv for exact "
                 "per-frame times). Drop this flight's .bbl into this session's "
                 "blackbox/ subfolder, then run combine_flight.py."),
        "config": {
            "channel_map": {"roll": channels.CH_ROLL, "pitch": channels.CH_PITCH,
                            "throttle": channels.CH_THR, "yaw": channels.CH_YAW,
                            "arm": channels.ARM_CH, "aux2_blackbox": channels.AUX2_CH,
                            "mode": channels.MODE_CH},
            "arm_us": {"armed": channels.ARM_ARMED_US,
                       "disarmed": channels.ARM_DISARMED_US},
            "aux2_us": {"high": channels.AUX2_HIGH_US, "mid": channels.AUX2_MID_US,
                        "low": channels.AUX2_LOW_US},
            "joystick_axis_map": {k: cal["axes"][k]["axis"]
                                  for k in ("roll", "pitch", "throttle", "yaw")}
                                 if cal and cal.get("axes") else {},
            "neutral_us": channels.NEUTRAL_US, "idle_thr_us": channels.IDLE_THR_US,
            "tx_hz": channels.TX_HZ,
            "ranger_port": port, "ranger_baud": baud,
            "camera": {"device_index": channels.DEVICE_INDEX,
                       "width": channels.WIDTH, "height": channels.HEIGHT},
        },
        "streams": {
            "commands": "commands.csv", "telemetry": "telemetry.csv",
            "telemetry_raw": "telemetry_raw.csv", "video": "video.mkv",
            "video_frames": "video_frames.csv", "vicon": "vicon.mat",
        },
    }
    if extra:
        meta.update(extra)
    if cmd_log is not None:
        meta["commands"] = {"file": "commands.csv", "rows": len(cmd_log.rows)}
    if telem is not None:
        meta["telemetry"] = {"file": "telemetry.csv", "rows": len(telem.typed),
                             "raw_file": "telemetry_raw.csv", "raw_rows": len(telem.raw)}
    if recorder is not None:
        ff = recorder.first_frame_wall
        meta["video"] = {
            "file": "video.mkv", "frames": recorder.frames, "fps": recorder.fps,
            "first_frame_wall": ff,
            "start_offset_s": (ff - session["t0"]) if ff else None,
            "status": recorder.status,
        }
    if vicon is not None:
        fp = vicon.first_packet_wall
        meta["vicon"] = {
            "file": "vicon.mat", "recorded": True, "samples": vicon.samples,
            "first_packet_wall": fp,
            "first_packet_offset_s": (fp - session["t0"]) if fp else None,
            "status": vicon.status,
        }
    else:
        # Record WHY pose is missing, so a no-Vicon flight explains itself later.
        meta["vicon"] = {"recorded": False,
                         "reason": vicon_off_reason or "vicon disabled"}
    with open(os.path.join(session["dir"], "session.json"), "w") as f:
        json.dump(meta, f, indent=2)


def warn_vicon_off(reason, udp_port=VICON_UDP_PORT):
    """Print a big, unmissable red banner when Vicon won't record — so a missing
    Vicon stream is obvious at launch instead of a one-liner that scrolls past."""
    bar = "!" * 70
    sys.stdout.write(
        f"\n{CSI}1;37;41m {bar} {CSI}0m\n"
        f"{CSI}1;31m  ⚠  VICON IS OFF — NO POSE / TRAJECTORY WILL BE RECORDED  ⚠{CSI}0m\n"
        f"{CSI}31m     reason: {reason or 'unknown'}{CSI}0m\n"
        f"{CSI}31m     is Vicon Tracker streaming to THIS laptop on UDP :{udp_port}? "
        f"start it, then restart this script.{CSI}0m\n"
        f"{CSI}1;37;41m {bar} {CSI}0m\n\n")
    sys.stdout.flush()
