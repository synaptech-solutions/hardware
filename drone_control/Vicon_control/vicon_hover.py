#!/usr/bin/env python3
"""Vicon autonomous hover: TX12 arms → drone takes off, climbs CLIMB_M, hovers.

The laptop reads LIVE Vicon pose and closes the loop, sending CRSF out the Ranger
the SAME way the data logger does. The TX12 stays in the loop for SAFETY + your
manual triggers only:
  - ARM switch (AUX3):  the master enable. Arm to begin; FLICK TO DISARM = instant
                        kill at any time. The controller never overrides this.
  - RECORD switch (AUX2): the launch + record trigger. With the drone armed,
                        flipping it HIGH captures the takeoff pose and commands the
                        climb to CLIMB_M; it also starts the synced recording (FC
                        blackbox + video + Vicon + commands + telemetry). Flip it
                        back to land gently. (Same switch, same recording, as the
                        data logger — these flights merge + render identically.)
The controller owns roll/pitch/throttle/yaw while flying and forces AUX4 = ANGLE
(it commands angle setpoints). The TX12 gimbals are ignored.

State machine (SINGLE FLIGHT — it arms, flies once, lands, and the program EXITS):
  DISARMED   — arm switch low. Idle + disarm. (Edge-gated: must see DISARMED once.)
  ARMED_IDLE — armed, motors idle (throttle ≤ arm threshold so the FC can arm).
               Waiting for the FC to confirm armed + the record switch to launch.
  FLYING     — climb to CLIMB_M and hold takeoff x/y + heading on Vicon.
  LANDING    — Vicon-held descent to LAND_CUT_M above launch, then CUT throttle +
               disarm + save the session + EXIT.
The flight ENDS (descend→cut→disarm→save→exit) on any of:
  - SPACEBAR on the laptop  → controlled landing.
  - Low battery             → controlled landing.
  - Disarm (TX12 arm low)   → instant kill.
  - Vicon loss              → blind sink if brief, else cut.
There is NO return to idle and NO time cap, so the controller can never
auto-relaunch (the 2026-06-11 land→takeoff loop is impossible by construction).

Dry-run (no Ranger, never arms) if EITHER --dry-run is passed OR config.DRY_RUN is
True (the default). Going LIVE requires config.DRY_RUN=False AND no --dry-run flag.
  .venv/bin/python drone_control/Vicon_control/vicon_hover.py --dry-run  # print control, never arm
  .venv/bin/python drone_control/Vicon_control/vicon_hover.py            # LIVE (only if config.DRY_RUN=False)
Calibrate the TX12 once with the data logger (shared cal file):
  .venv/bin/python data_logging/joystick_flight.py --calibrate
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
    T_FLIGHT_MODE, T_BATTERY, T_ATTITUDE, T_LINK_STATS, T_DEVICE_INFO,
    build_crsf_msp_v2_request, CrsfMspParser, decode_msp_raw_imu,
    convert_msp_raw_imu_units, MSP_RAW_IMU_CMD, T_MSP_RESP,
)
from common.recorders import (                           # noqa: E402
    VideoRecorder, ViconRecorder, CommandLogger, TelemetryLogger,
    write_session_json, warn_vicon_off, cv2, CV2_OK, VICON_OK, _VICON_ERR,
)
from Vicon_control import config                         # noqa: E402
from Vicon_control.vicon_source import ViconPoseSource   # noqa: E402
from Vicon_control.controller import ViconHoverController  # noqa: E402
from Vicon_control.mission import HoldMission              # noqa: E402

CSI = "\033["
REC_DIR = os.path.join(HERE, "flight_logs")
LAND_TIMEOUT_S = 10.0          # give up a stuck descent and idle after this
BLIND_DESC_BLEED_US = 60       # throttle below learned hover during a Vicon dropout


def is_armed(mode):
    """True iff the FC flight-mode string reports an armed, non-error mode."""
    return mode is not None and not mode.endswith("*") and not mode.startswith("!")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def render(state, ch, pose, ctl, tx_armed, fc_armed, record_on, pose_age,
           flight_mode, pack_v, vic_samples, recording, dry, mission_lbl=""):
    mode_lbl = "DRY-RUN" if dry else "LIVE"
    col = {"DISARMED": "0", "ARMED_IDLE": "33", "FLYING": "32", "LANDING": "36"}.get(state, "0")
    arm_txt = (f"{CSI}32mARM{CSI}0m" if tx_armed else "safe")
    fc_txt = "fcARM" if fc_armed else "fc-"
    if pose is not None:
        p = (f"xyz=({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f}) "
             f"yaw={pose['yaw']*57.2958:+6.1f}")
        vic = f"{CSI}32mvic{vic_samples}{CSI}0m" if recording else \
              (f"vic{pose_age*1000:.0f}ms" if pose_age < 9e8 else f"{CSI}1;31mVIC?{CSI}0m")
    else:
        p, vic = "xyz=(--)", f"{CSI}1;31mVICON:OFF{CSI}0m"
    c = (f"des(R{ctl['desired_roll_deg']:+.1f} P{ctl['desired_pitch_deg']:+.1f})deg "
         f"hov={ctl['hover_us']:.0f}" if ctl else "")
    v = f"{pack_v:.2f}V" if pack_v is not None else "—"
    ms = f"{CSI}35m{mission_lbl}{CSI}0m " if mission_lbl else ""
    sys.stdout.write(
        f"\r{CSI}K[{mode_lbl}] {CSI}{col}m{state:10s}{CSI}0m {arm_txt} {fc_txt} "
        f"rec{'ON' if record_on else '--'} {ms}| "
        f"R{ch[channels.CH_ROLL]:4d} P{ch[channels.CH_PITCH]:4d} "
        f"T{ch[channels.CH_THR]:4d} Y{ch[channels.CH_YAW]:4d} | {p} {c} | "
        f"{vic} FC:{flight_mode or '—'} {v}")
    sys.stdout.flush()


def run(args, make_mission=None):
    """Shared Vicon flight loop. make_mission is an optional callable(launch_tuple)
    -> Mission that provides the per-tick world setpoint (see mission.py); when
    None it flies a static HoldMission, reproducing the original hover behavior."""
    cal = load_cal(args.cal_file)
    js = Joystick(args.js)
    print(f"Joystick: {js.name}  ({args.js})")
    if cal.get("record") is None:
        sys.exit("Calibration has no RECORD switch — it's the launch+record trigger. "
                 "Re-run: data_logging/joystick_flight.py --calibrate")

    # Vicon is the feedback — REQUIRED (unlike the data logger, where it's optional).
    if not VICON_OK:
        sys.exit(f"Vicon deps missing ({_VICON_ERR}) — run with the repo .venv.")
    source = ViconPoseSource(yaw_offset_deg=config.VICON_YAW_OFFSET_DEG)
    print(f"Vicon:    probing UDP :{source.port} for a stream …")
    if not source.prepare():
        warn_vicon_off(source.status, source.port)
        sys.exit("Vicon is the control feedback — cannot fly without it. "
                 "Start Vicon Tracker (streaming to this laptop), then re-run.")
    print(f"Vicon:    {source.status}")

    controller = ViconHoverController()

    # Recorders — the SAME synced data pipeline as the data logger. The Vicon
    # recorder shares the control receiver (one socket on :51001).
    cmd_log = CommandLogger()
    telem = TelemetryLogger()
    recorder = (VideoRecorder(channels.DEVICE_INDEX, channels.WIDTH, channels.HEIGHT)
                if (CV2_OK and config.RECORD_VIDEO) else None)
    vicon_rec = ViconRecorder()
    vicon_rec.prepare(external_udp=source.udp)     # share the control receiver
    video_state = ("on" if config.RECORD_VIDEO else "OFF (config.RECORD_VIDEO=False)") \
        if CV2_OK else "OFF (no cv2)"
    print(f"Recording: flight_logs/<stamp>/ — video {video_state}"
          f", vicon {vicon_rec.status}, commands+telemetry")

    # Dry-run if EITHER --dry-run is passed OR config.DRY_RUN is set. config.DRY_RUN
    # defaults True (safe), so going LIVE requires config.DRY_RUN=False AND no flag.
    dry = args.dry_run or config.DRY_RUN
    ser = None
    ranger_port = None
    if not dry:
        ranger_port = args.port or autodetect_port()
        if not ranger_port:
            js.close()
            sys.exit("No Ranger serial port found. Plug in the Ranger USB-C, or "
                     "pass it explicitly: vicon_hover.py /dev/ttyACM0")
        ser = open_ranger(ranger_port, args.baud)
        print(f"Ranger:   {ranger_port} @ {args.baud} baud")
    else:
        print(f"{CSI}33mRanger:   DRY-RUN — no serial, never transmits, never arms. "
              f"Move the drone by hand and check the des(R,P)/throttle directions."
              f"{CSI}0m")

    print(f"\nArm is edge-gated (flip DISARMED once to enable). Climb target = "
          f"{config.CLIMB_M:.2f} m. Ctrl-C to stop.\n")

    parser = CrsfParser()
    crsf_msp_parser = CrsfMspParser()
    msp_imu_req = build_crsf_msp_v2_request(MSP_RAW_IMU_CMD)
    flight_mode = None
    pack_v = None
    bytes_rx = 0

    state = "DISARMED"
    seen_disarmed = False
    land_requested = False         # set by SPACEBAR (or low batt) → controlled land + exit
    launch = None                  # (x0, y0, z0, yaw0) captured at takeoff
    mission = None                 # target provider (HoldMission, or a WaypointMission)
    fly_t0 = None
    land_t0 = None

    period = 1.0 / config.TX_HZ
    nxt = time.monotonic()
    prev_mono = time.monotonic()
    last_ping = last_msp = last_render = 0.0
    PREVIEW_WIN = "drone feed — vicon hover"
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
                "kind": "vicon_hover", "climb_m": config.CLIMB_M,
                "target": ({"x": launch[0], "y": launch[1], "z": launch[2] + config.CLIMB_M,
                            "yaw_rad": launch[3]} if launch else None),
                "gains": {"kp_fwd": config.KP_FWD_DEG_PER_M, "kd_fwd": config.KD_FWD_DEG_PER_MPS,
                          "kp_lat": config.KP_LAT_DEG_PER_M, "kd_lat": config.KD_LAT_DEG_PER_MPS,
                          "kp_up": config.KP_UP, "kv_up": config.KV_UP_US_PER_MPS,
                          "ki_up": config.KI_UP_US_PER_M, "kp_yaw": config.KP_YAW_US_PER_RAD}},
                "mission": (mission.summary() if mission is not None else None)}
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
        ch[channels.AUX2_CH] = channels.AUX2_MID_US   # MID = nothing (never LOW=erase)
        ch[channels.MODE_CH] = channels.MODE_ANGLE_US
        return ch

    def send_disarm():
        if ser is None:
            return
        frame = build_rc_channels_packed(disarm_frame())
        for _ in range(5):
            ser.write(frame)
            time.sleep(0.01)

    # SPACEBAR = land. Read the laptop keyboard non-blocking in cbreak mode
    # (single keypress, no Enter; Ctrl-C still works since ISIG stays on).
    stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    stdin_old = termios.tcgetattr(stdin_fd) if stdin_fd is not None else None
    if stdin_fd is not None:
        tty.setcbreak(stdin_fd)
        print(f"{CSI}36mSPACEBAR = land (descend to {config.LAND_CUT_M:.2f} m, "
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

            # SPACEBAR while flying → request a controlled landing (ignored on the
            # ground so a stray press before takeoff can't immediately land it).
            if space_pressed() and state == "FLYING":
                land_requested = True

            # --- drain telemetry (decode for control gating + log every frame) ---
            sess_active = session["dir"] is not None
            if ser is not None and ser.in_waiting:
                chunk = ser.read(ser.in_waiting)
                bytes_rx += len(chunk)
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
                                "up_rssi_dbm": lk["up_rssi1_dBm"], "dn_rssi_dbm": lk["dn_rssi_dBm"],
                                "up_snr_db": lk["up_snr_dB"], "dn_snr_db": lk["dn_snr_dB"],
                                "rf_mode": lk["rf_mode"], "active_ant": lk["active_ant"],
                                "up_tx_pwr_idx": lk["up_tx_pwr_idx"]})
                    elif ftype == T_DEVICE_INFO and sess_active and len(payload) >= 2:
                        name = payload[2:].split(b"\x00", 1)[0].decode("ascii", errors="replace")
                        if name:
                            telem.log_typed(t_wall, "device_info",
                                            {"device_addr": f"0x{payload[1]:02X}", "device_name": name})
                    elif ftype == T_MSP_RESP:
                        for mcmd, mpl in crsf_msp_parser.feed_chunk(payload):
                            if mcmd != MSP_RAW_IMU_CMD:
                                continue
                            raw = decode_msp_raw_imu(mpl)
                            if raw and sess_active:
                                c = convert_msp_raw_imu_units(raw)
                                telem.log_typed(t_wall, "imu", {
                                    "imu_ax_g": c["ax_g"], "imu_ay_g": c["ay_g"], "imu_az_g": c["az_g"],
                                    "imu_gx_dps": c["gx_dps"], "imu_gy_dps": c["gy_dps"],
                                    "imu_gz_dps": c["gz_dps"], "imu_mag": c["mag_norm"]})

            # In DRY-RUN there's no telemetry, so trust the TX12 arm switch for
            # the FC-armed gate (lets you reach FLYING to check directions by hand).
            fc_armed = is_armed(flight_mode) if ser is not None else tx_armed

            pose = source.get_pose(now_wall)
            pose_age = pose["age_s"] if pose else 1e9
            pose_fresh = pose is not None and pose_age < config.VICON_STALE_S

            # --- battery cutoff (any state) ---
            batt_low = (config.LOW_BATT_CUTOFF and pack_v is not None
                        and config.BATT_PRESENT_V < pack_v < config.MIN_CELL_V * config.CELLS)

            # ===================== state transitions =====================
            # Single flight: any ending event descends/cuts, disarms, saves, EXITS.
            # Low battery in the air → request the same controlled landing as SPACEBAR.
            if batt_low and state in ("FLYING", "LANDING"):
                if not land_requested:
                    sys.stdout.write(f"\n{CSI}1;31mBATTERY LOW ({pack_v:.2f}V) — landing.{CSI}0m\n")
                land_requested = True

            if not tx_armed:
                # Manual disarm: in the air = instant kill + EXIT; on the ground =
                # stay disarmed, wait to be armed (edge-gated, never launches itself).
                if state in ("FLYING", "LANDING"):
                    sys.stdout.write(f"\n{CSI}1;31m✖ DISARM (TX12) — kill + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                if sess_active:
                    end_session()
                state, launch, mission, fly_t0, land_t0 = \
                    "DISARMED", None, None, None, None
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
                    launch = (pose["x"], pose["y"], pose["z"], pose["yaw"])
                    mission = (make_mission(launch) if make_mission is not None
                               else HoldMission(launch))
                    controller.set_target(launch[0], launch[1],
                                          launch[2] + config.CLIMB_M, launch[3])
                    begin_session()
                    fly_t0, state = now_mono, "FLYING"
                    sys.stdout.write(
                        f"\n{CSI}32m▶ LAUNCH{CSI}0m from ({launch[0]:+.2f},{launch[1]:+.2f},"
                        f"{launch[2]:+.2f}) → climb to {launch[2] + config.CLIMB_M:+.2f} m  "
                        f"(SPACEBAR to land)\n")
                    for line in mission.describe():
                        sys.stdout.write(line + "\n")
            elif state == "FLYING":
                if pose is None or pose_age > config.VICON_KILL_S:
                    sys.stdout.write(f"\n{CSI}1;31mVICON LOST {pose_age:.2f}s — cut + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                elif land_requested:
                    state, land_t0 = "LANDING", now_mono
                    sys.stdout.write(
                        f"\n{CSI}36m▼ LANDING — descend to {config.LAND_CUT_M:.2f} m, "
                        f"then cut + exit.{CSI}0m\n")
            elif state == "LANDING":
                z0 = launch[2] if launch else 0.0
                if pose is None or pose_age > config.VICON_KILL_S:
                    sys.stdout.write(f"\n{CSI}1;31mVICON LOST during land — cut + exit.{CSI}0m\n")
                    if sess_active:
                        end_session()
                    send_disarm()
                    break
                at_cut = pose["z"] <= z0 + config.LAND_CUT_M
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
            ch[channels.MODE_CH] = channels.MODE_ANGLE_US                 # controller needs ANGLE
            # AUX2: force HIGH while flying (FC blackbox covers the whole flight);
            # otherwise relay the switch so HIGH=start / LOW=erase still work on the ground.
            ch[channels.AUX2_CH] = (channels.AUX2_HIGH_US if state in ("FLYING", "LANDING")
                                    else aux2_us_for_switch(js, cal.get("record")))
            ctl_out = None
            if state == "FLYING" and pose_fresh:
                # Until airborne (TAKEOFF_AIRBORNE_M above the launch altitude), hold
                # level + freeze horizontal integrators so it lifts straight up.
                airborne = (launch is not None
                            and (pose["z"] - launch[2]) > config.TAKEOFF_AIRBORNE_M)
                # The mission supplies the world setpoint (a crawling carrot for the
                # waypoint course; a fixed point for HoldMission) AND its velocity
                # (D-term feedforward — pacing the carrot isn't braking-worthy).
                # set_setpoint keeps the learned hover throttle + integrators
                # (set_target would wipe them). done → land via the SAME path as
                # SPACEBAR.
                tx, ty, tz, tyaw, tvx, tvy, done = mission.update(pose, dt, airborne)
                controller.set_setpoint(tx, ty, tz, tyaw, tvx, tvy)
                ctl_out = controller.step(pose, dt, level_only=not airborne)
                if done and not land_requested:
                    land_requested = True
                    sys.stdout.write(f"\n{CSI}32m✔ MISSION COMPLETE — landing.{CSI}0m\n")
            elif state == "LANDING" and pose is not None:
                ctl_out = controller.step(pose, dt, descent_rate=config.LAND_SPEED_MPS)
            elif state == "FLYING" and pose is not None:
                # Stale Vicon (STALE < age < KILL): don't act on stale position —
                # hold level attitude and bleed throttle for a gentle blind sink.
                ch[channels.CH_THR] = int(clamp(controller.hover_us - BLIND_DESC_BLEED_US,
                                                channels.IDLE_THR_US, config.MAX_THROTTLE_US))
            if ctl_out is not None:
                ch[channels.CH_ROLL] = ctl_out["roll_us"]
                ch[channels.CH_PITCH] = ctl_out["pitch_us"]
                ch[channels.CH_YAW] = ctl_out["yaw_us"]
                ch[channels.CH_THR] = ctl_out["throttle_us"]
            ch = [int(clamp(c, 1000, 2000)) for c in ch]

            # --- log every commanded frame while a session is active ---
            if sess_active:
                cmd_log.log(now_wall, ch, tx_armed, record_on, js.axes, js.buttons)

            # --- transmit (+ keepalive ping + 10 Hz MSP IMU poll), like the logger ---
            if ser is not None:
                try:
                    ser.write(build_rc_channels_packed(ch))
                except Exception as e:
                    print(f"\n!! Ranger write failed: {e} — disarming.", flush=True)
                    break
                if now_mono - last_ping > 2.0:
                    ser.write(build_device_ping())
                    last_ping = now_mono
                if now_mono - last_msp > 0.1:
                    ser.write(msp_imu_req)
                    last_msp = now_mono

            # --- status line + optional preview (throttled, off the control path) ---
            if now_mono - last_render > 0.066:
                mission_lbl = mission.status() if mission is not None else ""
                render(state, ch, pose, ctl_out, tx_armed, fc_armed, record_on, pose_age,
                       flight_mode, pack_v, vicon_rec.samples, vicon_rec.recording, dry,
                       mission_lbl)
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
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_old)   # restore the terminal
        if session["dir"]:
            end_session()
        if window_open and CV2_OK:
            cv2.destroyAllWindows()
        if ser is not None:
            send_disarm()
            ser.close()
        js.close()
        source.stop()        # non-daemon Vicon receiver — stop it for a clean exit
        print("Stopped.")


def build_parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", default=None, help="Ranger serial port (default: autodetect)")
    ap.add_argument("baud", nargs="?", type=int, default=420000, help="Ranger baud (default: 420000)")
    ap.add_argument("--js", default="/dev/input/js0", help="joystick device")
    ap.add_argument("--cal-file", default=DEFAULT_CAL_PATH, help="TX12 calibration JSON (shared)")
    ap.add_argument("--dry-run", action="store_true",
                    help="read TX12 + Vicon, print control outputs; never open Ranger / never arm")
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
