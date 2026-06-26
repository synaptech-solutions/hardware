#!/usr/bin/env python3
"""Autonomous single-gate flight by a betaflight-gym recurrent vision policy — ACRO.

GateNet runs on the analog feed (its own thread) producing a mask; the recurrent
policy turns mask + onboard proprio into AETR commands out the Ranger. The TX12 is
SAFETY ONLY: ARM (AUX3) is the master enable / instant kill; RECORD (AUX2) HIGH
launches the policy + starts the synced recording.

The policy never sees position. Vicon only synthesises attitude/rates for the obs
(Betaflight doesn't stream gyro fast enough over CRSF) and backs the automated kill
(launch-relative box, Vicon loss). The gate policy has no landing behaviour, so a
flight ENDS by cut + disarm — fly low + cautious, finger on the disarm.

  .venv/bin/python drone_control/gate_control/gate_flight.py --dry-run --preview
"""
import argparse
import datetime
import os
import select
import sys
import termios
import time
import tty

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

# Pin JAX to CPU before any jax import (matches GateNet); JAX_PLATFORMS=cuda wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from common import channels                              # noqa: E402
from common.ranger import open_ranger                    # noqa: E402
from common.tx12 import (                                # noqa: E402
    Joystick, load_cal, arm_is_armed, record_switch_on, aux2_us_for_switch,
    DEFAULT_CAL_PATH,
)
from common.live_telemetry import (                      # noqa: E402
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    decode_flight_mode, decode_battery, decode_attitude, decode_link_stats,
    T_FLIGHT_MODE, T_BATTERY, T_ATTITUDE, T_LINK_STATS,
)
from common.recorders import (                           # noqa: E402
    ViconRecorder, CommandLogger, TelemetryLogger,
    write_session_json, warn_vicon_off, cv2, CV2_OK, VICON_OK, _VICON_ERR,
)
# shared Vicon UDP receiver (lives in Vicon_control)
from Vicon_control.vicon_source import ViconPoseSource   # noqa: E402
from gate_control import config                          # noqa: E402
from gate_control.gate_policy import GatePolicy, GatePolicyController  # noqa: E402
from gate_control.mask_source import MaskSource          # noqa: E402

CSI = "\033["
REC_DIR = os.path.join(HERE, "flight_logs")


def is_armed(mode):
    return mode is not None and not mode.endswith("*") and not mode.startswith("!")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def render(state, ch, pose, launch_w, tx_armed, fc_armed, record_on, pose_age,
           flight_mode, pack_v, vic_samples, recording, mask_age, mask_frames):
    col = {"DISARMED": "0", "ARMED_IDLE": "33", "FLYING": "32"}.get(state, "0")
    arm_txt = (f"{CSI}32mARM{CSI}0m" if tx_armed else "safe")
    fc_txt = "fcARM" if fc_armed else "fc-"
    if pose is not None:
        p = f"xyz=({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f})"
        vic = f"{CSI}32mvic{vic_samples}{CSI}0m" if recording else \
              (f"vic{pose_age*1000:.0f}ms" if pose_age < 9e8 else f"{CSI}1;31mVIC?{CSI}0m")
    else:
        p, vic = "xyz=(--)", f"{CSI}1;31mVICON:OFF{CSI}0m"
    if mask_frames == 0:
        m = f"{CSI}1;31mNO MASK{CSI}0m"
    elif mask_age > config.GATE_MASK_STALE_S:
        m = f"{CSI}33mmask{mask_age*1000:.0f}ms{CSI}0m"
    else:
        m = f"{CSI}32mmask{mask_age*1000:.0f}ms{CSI}0m"
    if pose is not None and launch_w is not None:
        r = ((pose["x"] - launch_w[0]) ** 2 + (pose["y"] - launch_w[1]) ** 2) ** 0.5
        c = f"r{r:.2f} dz{pose['z'] - launch_w[2]:+.2f}"   # launch-relative box state
    else:
        c = ""
    v = f"{pack_v:.2f}V" if pack_v is not None else "—"
    sys.stdout.write(
        f"\r{CSI}K[GATE] {CSI}{col}m{state:10s}{CSI}0m {arm_txt} {fc_txt} "
        f"rec{'ON' if record_on else '--'} ACRO {m} | "
        f"R{ch[channels.CH_ROLL]:4d} P{ch[channels.CH_PITCH]:4d} "
        f"T{ch[channels.CH_THR]:4d} Y{ch[channels.CH_YAW]:4d} | {p} {c} | "
        f"{vic} FC:{flight_mode or '—'} {v}")
    sys.stdout.flush()


def run(args):
    cal = load_cal(args.cal_file)
    js = Joystick(args.js)
    print(f"Joystick: {js.name}  ({args.js})")
    if cal.get("record") is None:
        sys.exit("Calibration has no RECORD switch — it's the launch+record trigger. "
                 "Re-run: data_logging/joystick_flight.py --calibrate")

    if not os.path.isfile(os.path.join(args.model, "gate_policy.json")):
        sys.exit(f"Gate policy bundle not found in: {args.model}\nExport it in betaflight-gym: "
                 f"python -m rl.export_recurrent <run_dir> -o {args.model}")
    policy = GatePolicy(args.model)
    print(f"Policy:   {policy.describe()}")

    if not VICON_OK:
        sys.exit(f"Vicon deps missing ({_VICON_ERR}) — run with the repo .venv.")
    # yaw_offset=0: the controller builds the full body frame from the raw quaternion.
    source = ViconPoseSource(yaw_offset_deg=0.0)
    print(f"Vicon:    probing UDP :{source.port} for a stream …")
    if not source.prepare():
        warn_vicon_off(source.status, source.port)
        sys.exit("Vicon is the proprio + safety source — cannot fly without it "
                 "(Betaflight doesn't stream gyro fast enough over CRSF).")
    print(f"Vicon:    {source.status}")

    controller = GatePolicyController(
        policy, mass_kg=args.mass, sign_roll=args.sign_roll,
        sign_pitch=args.sign_pitch, sign_yaw=args.sign_yaw)
    control_dt = policy.control_dt
    print(f"Safety:   launch-relative box  radius={config.MAX_RADIUS_M:.1f}m  "
          f"alt +{config.MAX_ALT_ABOVE_LAUNCH_M:.1f}/-{config.MAX_DESCENT_BELOW_LAUNCH_M:.1f}m  "
          f"(loop @ control_dt={control_dt*1000:.0f}ms = {1.0/control_dt:.0f}Hz)")

    # MaskSource owns the Cam Link device — no VideoRecorder on it here.
    mask_src = MaskSource(policy.mask_h, device=args.cam_device,
                          width=args.cam_width, height=args.cam_height,
                          fps=args.cam_fps, conf=config.GATE_MASK_CONF)
    print("Camera:   starting GateNet mask thread …")
    mask_src.start()
    print(f"Camera:   {mask_src.status} (mask {policy.mask_h}x{policy.mask_w})")

    cmd_log = CommandLogger()
    telem = TelemetryLogger()
    vicon_rec = ViconRecorder()
    vicon_rec.prepare(external_udp=source.udp)
    print(f"Recording: flight_logs/<stamp>/ — vicon {vicon_rec.status}, commands+telemetry "
          f"(VideoRecorder disabled to free the camera for the mask thread)")

    dry = args.dry_run or config.DRY_RUN
    ser = None
    ranger_port = None
    if not dry:
        ranger_port = args.port or autodetect_port()
        if not ranger_port:
            js.close()
            mask_src.stop()
            sys.exit("No Ranger serial port found. Plug in the Ranger USB-C, or pass it.")
        ser = open_ranger(ranger_port, args.baud)
        print(f"Ranger:   {ranger_port} @ {args.baud} baud")
    else:
        print(f"{CSI}33mRanger:   DRY-RUN — no serial, never transmits, never arms.{CSI}0m")

    print(f"\nFlight mode = ACRO. Arm is edge-gated (flip DISARMED once). The flight "
          f"ENDS by CUT+disarm (no policy landing) — fly low, finger on disarm. "
          f"Ctrl-C to stop.\n")

    parser = CrsfParser()
    flight_mode = None
    pack_v = None

    state = "DISARMED"
    seen_disarmed = False
    end_requested = False
    fly_t0 = None
    launch_w = None                # launch position (Vicon world), for the geofence

    period = control_dt
    nxt = time.monotonic()
    prev_mono = time.monotonic()
    last_ping = last_render = 0.0
    PREVIEW_WIN = "drone feed — gate policy"
    window_open = False

    session = {"dir": None, "t0": None, "stamp": None}

    def begin_session():
        t0 = time.time()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sdir = os.path.join(REC_DIR, stamp)
        os.makedirs(os.path.join(sdir, "blackbox"), exist_ok=True)
        session.update(dir=sdir, t0=t0, stamp=stamp)
        vicon_rec.start(t0, os.path.join(sdir, "vicon.mat"))
        cmd_log.start(t0, os.path.join(sdir, "commands.csv"), js.n_axes, js.n_buttons)
        telem.start(t0, os.path.join(sdir, "telemetry.csv"),
                    os.path.join(sdir, "telemetry_raw.csv"))
        sys.stdout.write(f"\n{CSI}32m● REC SESSION {stamp}{CSI}0m → {sdir}\n")

    def end_session():
        vicon_rec.stop()
        cmd_log.save()
        telem.save()
        if session["dir"]:
            extra = {"controller": {
                "kind": "gate_flight", "mode": "acro",
                "policy_dir": os.path.abspath(args.model),
                "policy_contract": policy.contract,
                "launch_w": None if launch_w is None else launch_w.tolist(),
                "max_radius_m": config.MAX_RADIUS_M,
                "max_alt_above_launch_m": config.MAX_ALT_ABOVE_LAUNCH_M,
                "signs": {"roll": args.sign_roll, "pitch": args.sign_pitch,
                          "yaw": args.sign_yaw}}}
            write_session_json(session, None, vicon_rec, cmd_log, telem, cal,
                               ranger_port, args.baud, extra=extra)
            sys.stdout.write(f"\n{CSI}33m■ SESSION SAVED{CSI}0m {session['stamp']}  "
                             f"vicon:{vicon_rec.status}\n"
                             f"   → drop the FC .bbl into {session['dir']}/blackbox/ "
                             f"then: Vicon_control/combine.py\n")
        session.update(dir=None, t0=None, stamp=None)

    def disarm_frame():
        ch = [channels.NEUTRAL_US] * 16
        ch[channels.CH_THR] = channels.IDLE_THR_US
        ch[channels.ARM_CH] = channels.ARM_DISARMED_US
        ch[channels.AUX2_CH] = channels.AUX2_MID_US
        ch[channels.MODE_CH] = channels.MODE_ACRO_US
        return ch

    def send_disarm():
        if ser is None:
            return
        frame = build_rc_channels_packed(disarm_frame())
        for _ in range(5):
            ser.write(frame)
            time.sleep(0.01)

    stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    stdin_old = termios.tcgetattr(stdin_fd) if stdin_fd is not None else None
    if stdin_fd is not None:
        tty.setcbreak(stdin_fd)
        print(f"{CSI}36mSPACEBAR = end flight (cut + disarm + save + exit).{CSI}0m")

    def space_pressed():
        if stdin_fd is None:
            return False
        hit = False
        while select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == " ":
                hit = True
        return hit

    def end_and_exit(reason):
        sys.stdout.write(f"\n{reason}\n")
        if session["dir"] is not None:
            end_session()
        send_disarm()

    try:
        while True:
            now_mono = time.monotonic()
            now_wall = time.time()
            dt = clamp(now_mono - prev_mono, 0.0, 0.05)
            prev_mono = now_mono

            js.poll()
            if not js.alive:
                print("\n!! joystick disconnected — disarming.", flush=True)
                break

            raw_armed = arm_is_armed(js, cal["arm"])
            if not raw_armed:
                seen_disarmed = True
            tx_armed = raw_armed and seen_disarmed
            record_on = record_switch_on(js, cal.get("record"))

            if space_pressed() and state == "FLYING":
                end_requested = True

            # --- drain telemetry (battery cutoff + blackbox-synced logging) ---
            sess_active = session["dir"] is not None
            if ser is not None and ser.in_waiting:
                chunk = ser.read(ser.in_waiting)
                for ftype, payload in parser.feed(chunk):
                    t_wall = time.time()
                    if sess_active:
                        telem.log_raw(t_wall, ftype, payload)
                    if ftype == T_FLIGHT_MODE:
                        flight_mode = decode_flight_mode(payload)
                        if sess_active:
                            telem.log_typed(t_wall, "flight_mode", {"flight_mode": flight_mode})
                    elif ftype == T_BATTERY:
                        b = decode_battery(payload)
                        if b:
                            pack_v = b["voltage_V"]
                            if sess_active:
                                telem.log_typed(t_wall, "battery", {
                                    "bat_v": b["voltage_V"], "bat_a": b["current_A"],
                                    "bat_mah": b["capacity_mAh"], "bat_pct": b["remaining_pct"]})
                    elif ftype == T_ATTITUDE and sess_active:
                        a = decode_attitude(payload)
                        if a:
                            telem.log_typed(t_wall, "attitude", {
                                "att_pitch_deg": a["pitch_deg"], "att_roll_deg": a["roll_deg"],
                                "att_yaw_deg": a["yaw_deg"]})
                    elif ftype == T_LINK_STATS and sess_active:
                        lk = decode_link_stats(payload)
                        if lk:
                            telem.log_typed(t_wall, "link", {
                                "up_lq": lk["up_lq"], "dn_lq": lk["dn_lq"],
                                "up_rssi_dbm": lk["up_rssi1_dBm"], "dn_rssi_dbm": lk["dn_rssi_dBm"]})

            fc_armed = is_armed(flight_mode) if ser is not None else tx_armed

            pose = source.get_pose(now_wall)
            pose_age = pose["age_s"] if pose else 1e9
            pose_ok = pose is not None and pose_age < config.VICON_KILL_S
            mask, mask_age = mask_src.get_mask(now_wall)

            batt_low = (config.LOW_BATT_CUTOFF and pack_v is not None
                        and config.BATT_PRESENT_V < pack_v < config.MIN_CELL_V * config.CELLS)

            # ===================== state transitions =====================
            if not tx_armed:
                if state == "FLYING":
                    end_and_exit(f"{CSI}1;31m✖ DISARM (TX12) — kill + exit.{CSI}0m")
                    break
                if sess_active:
                    end_session()
                state, fly_t0, launch_w = "DISARMED", None, None
                end_requested = False
                controller.reset()
            elif state == "DISARMED":
                state = "ARMED_IDLE"
                controller.reset()
            elif state == "ARMED_IDLE":
                if batt_low:
                    end_and_exit(f"{CSI}1;31mBATTERY LOW ({pack_v:.2f}V) on the ground — exit.{CSI}0m")
                    break
                if fc_armed and record_on and pose_ok:
                    if mask_src.frames == 0:
                        if now_mono - last_render > 0.5:
                            sys.stdout.write(f"\r{CSI}K{CSI}1;31mwaiting for GateNet mask "
                                             f"before launch …{CSI}0m")
                            sys.stdout.flush()
                            last_render = now_mono
                    else:
                        controller.capture_launch(pose)
                        launch_w = controller.launch_w
                        begin_session()
                        fly_t0, state = now_mono, "FLYING"
                        sys.stdout.write(
                            f"\n{CSI}32m▶ LAUNCH{CSI}0m (gate policy) from "
                            f"({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f})  "
                            f"(SPACEBAR to end)\n")
            elif state == "FLYING":
                if pose is None or pose_age > config.VICON_KILL_S:
                    end_and_exit(f"{CSI}1;31m✖ VICON LOST {pose_age:.2f}s (out of range) — "
                                 f"cut + exit.{CSI}0m")
                    break
                r = ((pose["x"] - launch_w[0]) ** 2 + (pose["y"] - launch_w[1]) ** 2) ** 0.5
                dz = pose["z"] - launch_w[2]
                if r > config.MAX_RADIUS_M:
                    end_and_exit(f"{CSI}1;31m✖ GEOFENCE — {r:.1f}m from launch "
                                 f"(> {config.MAX_RADIUS_M:.1f}) — cut + exit.{CSI}0m")
                    break
                if dz > config.MAX_ALT_ABOVE_LAUNCH_M:
                    end_and_exit(f"{CSI}1;31m✖ CEILING — {dz:.1f}m above launch "
                                 f"(> {config.MAX_ALT_ABOVE_LAUNCH_M:.1f}) — cut + exit.{CSI}0m")
                    break
                if dz < -config.MAX_DESCENT_BELOW_LAUNCH_M:
                    end_and_exit(f"{CSI}1;31m✖ FLOOR — {-dz:.1f}m below launch "
                                 f"(> {config.MAX_DESCENT_BELOW_LAUNCH_M:.1f}) — cut + exit.{CSI}0m")
                    break
                if batt_low:
                    end_and_exit(f"{CSI}1;31mBATTERY LOW ({pack_v:.2f}V) — cut + exit.{CSI}0m")
                    break
                if end_requested:
                    end_and_exit(f"{CSI}36m■ END (SPACEBAR) — cut + disarm + exit.{CSI}0m")
                    break

            # ===================== build output frame =====================
            ch = [channels.NEUTRAL_US] * 16
            ch[channels.CH_THR] = channels.IDLE_THR_US
            ch[channels.ARM_CH] = channels.ARM_ARMED_US if tx_armed else channels.ARM_DISARMED_US
            ch[channels.MODE_CH] = channels.MODE_ACRO_US
            ch[channels.AUX2_CH] = (channels.AUX2_HIGH_US if state == "FLYING"
                                    else aux2_us_for_switch(js, cal.get("record")))
            if state == "FLYING":
                # pose is fresh here: the Vicon-loss kill above already fired otherwise
                us = controller.step(pose, mask, dt)["us"]
                ch[channels.CH_ROLL] = int(us[0])
                ch[channels.CH_PITCH] = int(us[1])
                ch[channels.CH_THR] = int(us[2])
                ch[channels.CH_YAW] = int(us[3])
            ch = [int(clamp(c, 1000, 2000)) for c in ch]

            if sess_active:
                cmd_log.log(now_wall, ch, tx_armed, record_on, js.axes, js.buttons)

            # --- transmit (+ keepalive ping) ---
            if ser is not None:
                try:
                    ser.write(build_rc_channels_packed(ch))
                except Exception as e:
                    print(f"\n!! Ranger write failed: {e} — disarming.", flush=True)
                    break
                if now_mono - last_ping > 2.0:
                    ser.write(build_device_ping())
                    last_ping = now_mono

            if now_mono - last_render > 0.066:
                render(state, ch, pose, launch_w, tx_armed, fc_armed, record_on, pose_age,
                       flight_mode, pack_v, vicon_rec.samples, vicon_rec.recording,
                       mask_age, mask_src.frames)
                if CV2_OK and args.preview:
                    frame = mask_src.get_latest_frame()
                    if frame is not None:
                        disp = frame.copy()
                        m = (cv2.resize((mask * 255).astype("uint8"),
                                        (disp.shape[1], disp.shape[0])) > 96)
                        disp[m] = (0.5 * disp[m] + (0, 140, 0)).astype("uint8")
                        cv2.putText(disp, f"{state} T{ch[channels.CH_THR]} {mask_src.infer_ms:.0f}ms",
                                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(PREVIEW_WIN, disp)
                        cv2.waitKey(1)
                        window_open = True
                last_render = now_mono

            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        print("\nCtrl-C — disarming.", flush=True)
    finally:
        if stdin_old is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_old)
        if session["dir"]:
            end_session()
        if window_open and CV2_OK:
            cv2.destroyAllWindows()
        if ser is not None:
            send_disarm()
            ser.close()
        mask_src.stop()
        js.close()
        source.stop()
        print("Stopped.")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", default=None, help="Ranger serial port (default: autodetect)")
    ap.add_argument("baud", nargs="?", type=int, default=420000, help="Ranger baud (default: 420000)")
    ap.add_argument("--model", default=config.GATE_MODEL_DIR,
                    help="gate policy bundle dir (gate_actor.exp + gate_policy.json)")
    ap.add_argument("--mass", type=float, default=None,
                    help="measured all-up mass (kg); default = the policy's trained mass_kg")
    ap.add_argument("--js", default="/dev/input/js0", help="joystick device")
    ap.add_argument("--cal-file", default=DEFAULT_CAL_PATH, help="TX12 calibration JSON (shared)")
    ap.add_argument("--cam-device", type=int, default=channels.DEVICE_INDEX,
                    help="camera /dev/videoN for the GateNet feed")
    ap.add_argument("--cam-width", type=int, default=720, help="camera capture width (NTSC 720x480)")
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--preview", action="store_true",
                    help="show the drone feed with the mask overlaid (off the control path)")
    ap.add_argument("--sign-roll", type=int, default=1, choices=(1, -1),
                    help="flip if DRY-RUN shows roll commanded the wrong way")
    ap.add_argument("--sign-pitch", type=int, default=1, choices=(1, -1))
    ap.add_argument("--sign-yaw", type=int, default=1, choices=(1, -1))
    ap.add_argument("--dry-run", action="store_true",
                    help="read TX12 + Vicon + camera, print control; never open Ranger / arm")
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
