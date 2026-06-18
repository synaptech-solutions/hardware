#!/usr/bin/env python3
"""
Vicon autonomous hover flown by a trained betaflight-gym RL policy — ACRO.
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
    VideoRecorder, ViconRecorder, CommandLogger, TelemetryLogger,
    write_session_json, warn_vicon_off, cv2, CV2_OK, VICON_OK, _VICON_ERR,
)
from Vicon_control import config                         # noqa: E402
from Vicon_control.vicon_source import ViconPoseSource   # noqa: E402
from Vicon_control.rl_policy import MLPPolicy, HoverPolicyController, AxisMap  # noqa: E402

CSI = "\033["
REC_DIR = os.path.join(HERE, "flight_logs")
LAND_TIMEOUT_S = 10.0


def is_armed(mode):
    return mode is not None and not mode.endswith("*") and not mode.startswith("!")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def render(state, ch, pose, ctl, tx_armed, fc_armed, record_on, pose_age,
           flight_mode, pack_v, vic_samples, recording):
    col = {"DISARMED": "0", "ARMED_IDLE": "33", "FLYING": "32", "LANDING": "36"}.get(state, "0")
    arm_txt = (f"{CSI}32mARM{CSI}0m" if tx_armed else "safe")
    fc_txt = "fcARM" if fc_armed else "fc-"
    if pose is not None:
        p = (f"xyz=({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f})")
        vic = f"{CSI}32mvic{vic_samples}{CSI}0m" if recording else \
              (f"vic{pose_age*1000:.0f}ms" if pose_age < 9e8 else f"{CSI}1;31mVIC?{CSI}0m")
    else:
        p, vic = "xyz=(--)", f"{CSI}1;31mVICON:OFF{CSI}0m"
    if ctl is not None:
        pe, ti = ctl["pos_err"], ctl["tilt"]
        c = (f"err(f{pe[0]:+.2f} r{pe[1]:+.2f} d{pe[2]:+.2f}) "
             f"tilt(f{ti[0]:+.2f} r{ti[1]:+.2f})")
    else:
        c = ""
    v = f"{pack_v:.2f}V" if pack_v is not None else "—"
    sys.stdout.write(
        f"\r{CSI}K[RL] {CSI}{col}m{state:10s}{CSI}0m {arm_txt} {fc_txt} "
        f"rec{'ON' if record_on else '--'} ACRO | "
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

    if not os.path.isfile(args.policy):
        sys.exit(f"Policy not found: {args.policy}\nTrain + export it in betaflight-gym "
                 f"(scripts/train.py writes <run>/policy.npz).")
    policy = MLPPolicy(args.policy)
    print(f"Policy:   {policy.describe()}")
    if policy.obs_dim not in (22, 23):
        print(f"{CSI}1;33mWARNING obs_dim={policy.obs_dim} (expected 22, or 23 "
              f"for the mass-conditioned hover task) — frames/layout may not match.{CSI}0m")

    if not VICON_OK:
        sys.exit(f"Vicon deps missing ({_VICON_ERR}) — run with the repo .venv.")
    # yaw_offset=0: the policy reads the RAW quaternion via rl_policy.AxisMap (it
    # builds the full body frame itself), so the PID's yaw offset is irrelevant.
    source = ViconPoseSource(yaw_offset_deg=0.0)
    print(f"Vicon:    probing UDP :{source.port} for a stream …")
    if not source.prepare():
        warn_vicon_off(source.status, source.port)
        sys.exit("Vicon is the control feedback — cannot fly without it.")
    print(f"Vicon:    {source.status}")

    control_dt = float(policy.meta.get("control_dt", 1.0 / config.TX_HZ))
    target_alt = args.target_alt if args.target_alt is not None else None
    controller = HoverPolicyController(
        policy, axis_map=AxisMap(), target_alt=target_alt, control_dt=control_dt,
        sign_roll=args.sign_roll, sign_pitch=args.sign_pitch, sign_yaw=args.sign_yaw,
        land_speed_mps=config.LAND_SPEED_MPS, land_cut_m=config.LAND_CUT_M)
    print(f"Hover:    target_alt={controller.target_alt:.2f} m above launch  "
          f"(loop @ trained control_dt={control_dt*1000:.0f} ms = {1.0/control_dt:.0f} Hz)")

    cmd_log = CommandLogger()
    telem = TelemetryLogger()
    recorder = (VideoRecorder(channels.DEVICE_INDEX, channels.WIDTH, channels.HEIGHT)
                if (CV2_OK and config.RECORD_VIDEO) else None)
    vicon_rec = ViconRecorder()
    vicon_rec.prepare(external_udp=source.udp)
    video_state = ("on" if config.RECORD_VIDEO else "OFF (config.RECORD_VIDEO=False)") \
        if CV2_OK else "OFF (no cv2)"
    print(f"Recording: flight_logs/<stamp>/ — video {video_state}"
          f", vicon {vicon_rec.status}, commands+telemetry")

    ranger_port = args.port or autodetect_port()
    if not ranger_port:
        js.close()
        sys.exit("No Ranger serial port found. Plug in the Ranger USB-C, or "
                 "pass it explicitly.")
    ser = open_ranger(ranger_port, args.baud)
    print(f"Ranger:   {ranger_port} @ {args.baud} baud")

    print(f"\nArm is edge-gated (flip DISARMED once to enable). Flight mode = ACRO. "
          f"Ctrl-C to stop.\n")

    parser = CrsfParser()
    flight_mode = None
    pack_v = None

    state = "DISARMED"
    seen_disarmed = False
    land_requested = False
    launch_z = None
    airborne = False
    fly_t0 = land_t0 = None
    last_ch = None

    period = control_dt
    nxt = time.monotonic()
    prev_mono = time.monotonic()
    last_ping = last_render = 0.0
    PREVIEW_WIN = "drone feed — vicon RL hover"
    window_open = False

    session = {"dir": None, "t0": None, "stamp": None}

    def begin_session():
        t0 = time.time()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sdir = os.path.join(REC_DIR, stamp)
        os.makedirs(os.path.join(sdir, "blackbox"), exist_ok=True)
        session.update(dir=sdir, t0=t0, stamp=stamp)
        if recorder is not None:
            recorder.start(t0, os.path.join(sdir, "video.mkv"))
        vicon_rec.start(t0, os.path.join(sdir, "vicon.mat"))
        cmd_log.start(t0, os.path.join(sdir, "commands.csv"), js.n_axes, js.n_buttons)
        telem.start(t0, os.path.join(sdir, "telemetry.csv"),
                    os.path.join(sdir, "telemetry_raw.csv"))
        sys.stdout.write(f"\n{CSI}32m● REC SESSION {stamp}{CSI}0m → {sdir}\n")

    def end_session():
        if recorder is not None:
            recorder.stop()
        vicon_rec.stop()
        cmd_log.save()
        telem.save()
        if session["dir"]:
            extra = {"controller": {
                "kind": "vicon_rl_hover", "mode": "acro",
                "policy": os.path.abspath(args.policy),
                "policy_meta": policy.meta,
                "target_alt_m": controller.target_alt,
                "signs": {"roll": args.sign_roll, "pitch": args.sign_pitch,
                          "yaw": args.sign_yaw}}}
            write_session_json(session, recorder, vicon_rec, cmd_log, telem, cal,
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
        print(f"{CSI}36mSPACEBAR = land (policy descends to {config.LAND_CUT_M:.2f} m, "
              f"cut, disarm, save, exit).{CSI}0m")

    def space_pressed():
        if stdin_fd is None:
            return False
        hit = False
        while select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == " ":
                hit = True
        return hit

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
                land_requested = True

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

            fc_armed = is_armed(flight_mode)

            pose = source.get_pose(now_wall)
            pose_age = pose["age_s"] if pose else 1e9
            pose_fresh = pose is not None and pose_age < config.VICON_STALE_S

            batt_low = (config.LOW_BATT_CUTOFF and pack_v is not None
                        and config.BATT_PRESENT_V < pack_v < config.MIN_CELL_V * config.CELLS)

            # ===================== state transitions =====================
            if batt_low and state in ("FLYING", "LANDING"):
                if not land_requested:
                    sys.stdout.write(f"\n{CSI}1;31mBATTERY LOW ({pack_v:.2f}V) — landing.{CSI}0m\n")
                land_requested = True

            if not tx_armed:
                if state in ("FLYING", "LANDING"):
                    sys.stdout.write(f"\n{CSI}1;31m✖ DISARM (TX12) — kill + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                if sess_active:
                    end_session()
                state, launch_z, airborne, fly_t0, land_t0 = \
                    "DISARMED", None, False, None, None
                land_requested = False
                controller.reset()
            elif state == "DISARMED":
                state = "ARMED_IDLE"
                controller.reset()
            elif state == "ARMED_IDLE":
                if batt_low:
                    sys.stdout.write(f"\n{CSI}1;31mBATTERY LOW ({pack_v:.2f}V) on the ground — exit.{CSI}0m\n")
                    send_disarm()
                    break
                if fc_armed and record_on and pose_fresh:
                    controller.capture_launch(pose)
                    launch_z = pose["z"]
                    airborne = False
                    begin_session()
                    fly_t0, state = now_mono, "FLYING"
                    sys.stdout.write(
                        f"\n{CSI}32m▶ LAUNCH{CSI}0m (policy) from "
                        f"({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f}) → hold "
                        f"{controller.target_alt:.2f} m up  (SPACEBAR to land)\n")
            elif state == "FLYING":
                if pose is None or pose_age > config.VICON_KILL_S:
                    sys.stdout.write(f"\n{CSI}1;31mVICON LOST {pose_age:.2f}s — cut + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                if land_requested:
                    controller.begin_landing()
                    state, land_t0 = "LANDING", now_mono
                    sys.stdout.write(
                        f"\n{CSI}36m▼ LANDING — policy descends to {config.LAND_CUT_M:.2f} m, "
                        f"then cut + exit.{CSI}0m\n")
            elif state == "LANDING":
                if pose is None or pose_age > config.VICON_KILL_S:
                    sys.stdout.write(f"\n{CSI}1;31mVICON LOST during land — cut + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                at_cut = (launch_z is not None) and pose["z"] <= launch_z + config.LAND_CUT_M
                if at_cut or (now_mono - land_t0) > LAND_TIMEOUT_S:
                    why = "reached cut height" if at_cut else "land timeout"
                    sys.stdout.write(
                        f"\n{CSI}33m■ {why} (z={pose['z']:.2f} m) — cut throttle, "
                        f"disarm, save, exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break

            # ===================== build output frame =====================
            ch = [channels.NEUTRAL_US] * 16
            ch[channels.CH_THR] = channels.IDLE_THR_US
            ch[channels.ARM_CH] = channels.ARM_ARMED_US if tx_armed else channels.ARM_DISARMED_US
            ch[channels.MODE_CH] = channels.MODE_ACRO_US                  # ACRO the whole time
            ch[channels.AUX2_CH] = (channels.AUX2_HIGH_US if state in ("FLYING", "LANDING")
                                    else aux2_us_for_switch(js, cal.get("record")))
            ctl_out = None
            if state in ("FLYING", "LANDING"):
                if pose_fresh:
                    if launch_z is not None and (pose["z"] - launch_z) > config.TAKEOFF_AIRBORNE_M:
                        airborne = True
                    ctl_out = controller.step(pose, dt)
                    us = ctl_out["us"]
                    ch[channels.CH_ROLL] = int(us[0])
                    ch[channels.CH_PITCH] = int(us[1])
                    ch[channels.CH_THR] = int(us[2])
                    ch[channels.CH_YAW] = int(us[3])
                elif last_ch is not None:
                    # brief stale Vicon (STALE < age < KILL): repeat the last good
                    # command rather than act on a stale pose; KILL above cuts it.
                    ch = list(last_ch)
                    ch[channels.ARM_CH] = channels.ARM_ARMED_US if tx_armed else channels.ARM_DISARMED_US
            ch = [int(clamp(c, 1000, 2000)) for c in ch]
            if state in ("FLYING", "LANDING"):
                last_ch = list(ch)

            if sess_active:
                cmd_log.log(now_wall, ch, tx_armed, record_on, js.axes, js.buttons)

            # --- transmit (+ keepalive ping), like the logger ---
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
                render(state, ch, pose, ctl_out, tx_armed, fc_armed, record_on, pose_age,
                       flight_mode, pack_v, vicon_rec.samples, vicon_rec.recording)
                if recorder is not None and CV2_OK:
                    if recorder.recording:
                        frame = recorder.get_latest_frame()
                        if frame is not None:
                            disp = frame.copy()
                            cv2.putText(disp, f"{state}  T{ch[channels.CH_THR]}  vic{vicon_rec.samples}",
                                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            cv2.imshow(PREVIEW_WIN, disp)
                            cv2.waitKey(1)
                            window_open = True
                    elif window_open:
                        cv2.destroyAllWindows()
                        window_open = False
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
        js.close()
        source.stop()
        print("Stopped.")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", default=None, help="Ranger serial port (default: autodetect)")
    ap.add_argument("baud", nargs="?", type=int, default=420000, help="Ranger baud (default: 420000)")
    ap.add_argument("--policy", default=os.path.join(HERE, "models", "hover_acro.npz"),
                    help="trained policy.npz exported by betaflight-gym rl/export.py "
                         "(default: the bundled models/hover_acro.npz)")
    ap.add_argument("--target-alt", type=float, default=None,
                    help="hover altitude above launch (m); default = the policy's trained target_alt")
    ap.add_argument("--js", default="/dev/input/js0", help="joystick device")
    ap.add_argument("--cal-file", default=DEFAULT_CAL_PATH, help="TX12 calibration JSON (shared)")
    ap.add_argument("--sign-roll", type=int, default=1, choices=(1, -1),
                    help="flip if the first hover shows roll commanded the wrong way")
    ap.add_argument("--sign-pitch", type=int, default=1, choices=(1, -1))
    ap.add_argument("--sign-yaw", type=int, default=1, choices=(1, -1))
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
