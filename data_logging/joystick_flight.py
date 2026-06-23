#!/usr/bin/env python3
"""Manual flight: TX12 USB joystick → laptop → Ranger → drone (transparent RC relay).

The TX12 is in EdgeTX "USB Joystick (HID)" mode, so the laptop sees it as
/dev/input/js0 (7 axes, 24 buttons on this radio). This script reads the
gimbals + arm switch from that joystick and re-emits them as a CRSF
RC_CHANNELS stream out the Ranger's USB-C serial — the same send path the
autonomous controllers use. Net effect: you fly the drone by hand through the
laptop, exactly as if the TX12 were transmitting directly.

The reusable machinery now lives in the shared drone_control/common/ layer:
  common.tx12        — Joystick reader, calibration, switch interpreters
  common.channels    — Air75 CRSF channel map + µs levels (single source of truth)
  common.live_telemetry — CRSF/MSP frame builders + parsers
  common.ranger      — open_ranger() custom-baud open
  common.recorders   — the synced video/Vicon/command/telemetry recorders
This script is the thin orchestrator that wires them into the manual-flight loop;
the Vicon controller reuses the same common/ modules.

WHY CALIBRATION: EdgeTX's channel→USB-axis assignment is configurable and
radio-specific, so which HID axis is roll vs pitch vs throttle vs yaw — and
which control is the arm switch — is NOT safe to hardcode. `--calibrate`
discovers it empirically and writes tx12_joystick_cal.json (next to this script).

This FC has no prearm — a single arm switch gates arming (CH_ARM only).

RECORDING: the blackbox switch (relayed to AUX2 so the FC's blackbox starts)
also triggers laptop capture of EVERYTHING flowing through the laptop, all
stamped on one clock (t_rel = wall - t0) so they merge deterministically:
  - commands.csv      every outgoing RC frame: all 16 channels + full joystick
  - telemetry.csv     typed decode of incoming CRSF telemetry (attitude, battery,
                      link stats, flight mode, device info) + live IMU accel/gyro
                      (MSP_RAW_IMU, actively polled at 10 Hz)
  - telemetry_raw.csv every incoming frame as hex (lossless safety net)
  - video.mkv + video_frames.csv   drone feed (H.264/libx264 crf18) + per-frame capture timestamps
  - vicon.mat         Vicon pose @ 100 Hz
Recording runs while the drone is ARMED and the switch is ON; everything is saved
when either drops (disarm or switch off). Video/vicon need cv2/scipy — run with
the repo venv: .venv/bin/python. The CSV logs and FC blackbox have no such deps.

Usage:
    .venv/bin/python joystick_flight.py --calibrate  # one-time, per radio config
    .venv/bin/python joystick_flight.py --calibrate-mode  # add the flight-mode switch (AUX4) to an existing cal
    .venv/bin/python joystick_flight.py --dry-run    # read joystick, print sticks, never send
    .venv/bin/python joystick_flight.py              # fly: autodetect Ranger port
    .venv/bin/python joystick_flight.py /dev/ttyACM0 420000  # explicit Ranger port + baud

SAFETY
  - Starts disarmed, throttle idle.
  - Arm is edge-gated: the script refuses to pass ARM-high until it has first
    seen the arm switch in the DISARMED position (no arm-on-startup).
  - Joystick unplug / read error → forces disarm + idle and exits.
  - Ctrl-C → sends explicit disarm frames, then exits.
  - --dry-run never opens the Ranger and never sends anything.
"""
import argparse
import datetime
import os
import sys
import time

# Shared hardware/IO layer lives in the sibling drone_control/common/ package.
HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.join(os.path.dirname(HERE), "drone_control")
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from common import channels as config             # noqa: E402 — kept-code alias
from common.ranger import open_ranger             # noqa: E402
from common.tx12 import (                         # noqa: E402
    Joystick, load_cal, calibrate, calibrate_mode_only,
    map_axis, arm_is_armed, record_switch_on, aux2_us_for_switch,
    mode_us_for_switch, DEFAULT_CAL_PATH,
)
from common.live_telemetry import (               # noqa: E402
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    decode_flight_mode, decode_battery, decode_attitude, decode_link_stats,
    T_FLIGHT_MODE, T_BATTERY, T_ATTITUDE, T_LINK_STATS, T_DEVICE_INFO,
    build_crsf_msp_v2_request, CrsfMspParser, decode_msp_raw_imu,
    convert_msp_raw_imu_units, MSP_RAW_IMU_CMD, T_MSP_RESP,
)
from common.recorders import (                    # noqa: E402
    VideoRecorder, ViconRecorder, CommandLogger, TelemetryLogger,
    write_session_json as _write_session_json, warn_vicon_off as _warn_vicon_off,
    cv2, CV2_OK, VICON_OK, _VICON_ERR, _VICON_PROBE_TIMEOUT_S,
)

# Air75 channel map + µs levels — imported as bare names so the run loop below
# stays in lockstep with the single source of truth (common.channels).
from common.channels import (                      # noqa: E402
    ARM_THROTTLE_MAX_US, ARM_CH, ARM_ARMED_US, ARM_DISARMED_US,
    AUX2_CH, AUX2_MID_US, MODE_CH,
)

CAL_FILE_DEFAULT = DEFAULT_CAL_PATH
REC_DIR = os.path.join(HERE, "recordings")


def build_channels(js, cal, allow_arm):
    """Read joystick state → (16-channel µs list, armed flag, record_on flag).

    This FC has NO prearm — a single arm switch on AUX3 (ARM_CH) gates arming.
    Gimbals are the standard AETR layout on indices 0-3. The blackbox switch is
    relayed to AUX2 (AUX2_CH) so the FC's blackbox starts when it's high. The
    3-position flight-mode switch drives AUX4 (MODE_CH) → acro/angle/horizon."""
    ch = [config.NEUTRAL_US] * 16
    ax = cal["axes"]
    ch[config.CH_ROLL] = map_axis(js.axes.get(ax["roll"]["axis"], 0), ax["roll"])
    ch[config.CH_PITCH] = map_axis(js.axes.get(ax["pitch"]["axis"], 0), ax["pitch"])
    ch[config.CH_YAW] = map_axis(js.axes.get(ax["yaw"]["axis"], 0), ax["yaw"])
    ch[config.CH_THR] = map_axis(
        js.axes.get(ax["throttle"]["axis"], ax["throttle"]["lo_raw"]), ax["throttle"])

    armed = arm_is_armed(js, cal["arm"]) and allow_arm
    ch[ARM_CH] = ARM_ARMED_US if armed else ARM_DISARMED_US

    record_on = record_switch_on(js, cal.get("record"))
    ch[AUX2_CH] = aux2_us_for_switch(js, cal.get("record"))

    # AUX4 flight mode: acro / angle / horizon from the 3-position mode switch
    # (defaults to ACRO when uncalibrated — per user request).
    ch[MODE_CH] = mode_us_for_switch(js, cal.get("mode"))
    return ch, armed, record_on


# ===================== run loop =====================

CSI = "\033["


def render(ch, armed, allow_arm, raw_armed, record_on, recorder, vicon,
           flight_mode, pack_v, bytes_rx, dry):
    arm_txt = (f"{CSI}32mARMED{CSI}0m" if armed
               else (f"{CSI}33msafe (switch ARMED — toggle to DISARMED first){CSI}0m"
                     if raw_armed and not allow_arm else "safe"))
    fm = flight_mode if flight_mode is not None else "—"
    v = f"{pack_v:.2f}V" if pack_v is not None else "—"
    mode_lbl = "DRY-RUN (not sending)" if dry else "LIVE → Ranger"
    # When the arm switch is engaged but throttle is too high, Betaflight will
    # silently refuse to arm (THROTTLE flag) — call it out explicitly.
    thr_hint = ""
    if raw_armed and ch[config.CH_THR] > ARM_THROTTLE_MAX_US:
        thr_hint = (f" {CSI}31m[THROTTLE {ch[config.CH_THR]}us > {ARM_THROTTLE_MAX_US}"
                    f" — lower stick fully to arm]{CSI}0m")
    if record_on:
        vid = f"vid{recorder.frames}" if (recorder and recorder.recording) else "vid-"
        vic = f"vic{vicon.samples}" if (vicon and vicon.recording) else "vic-"
        rec_txt = f"{CSI}31m●REC{CSI}0m({vid},{vic})" if (armed) else \
                  f"{CSI}33mBB-on(arm to record){CSI}0m"
    else:
        rec_txt = "rec-off"
    # Flight mode the AUX4 value selects (acro/angle/horizon).
    mu = ch[MODE_CH]
    fmode = "HORIZON" if mu >= 1700 else ("ANGLE" if 1400 <= mu <= 1600 else "ACRO")
    # Persistent Vicon status — ALWAYS visible so a dead Vicon is obvious live.
    #   OFF (red, disabled) · ready (yellow) · ●N (green, recording N samples).
    if vicon is None:
        vic_txt = f"{CSI}1;31mVICON:OFF{CSI}0m"
    elif vicon.recording:
        vic_txt = f"{CSI}32mVICON:●{vicon.samples}{CSI}0m"
    else:
        vic_txt = f"{CSI}33mVICON:ready{CSI}0m"
    sys.stdout.write(
        f"\r{CSI}K[{mode_lbl}] "
        f"R{ch[config.CH_ROLL]:4d} P{ch[config.CH_PITCH]:4d} "
        f"T{ch[config.CH_THR]:4d} Y{ch[config.CH_YAW]:4d} | {arm_txt} "
        f"AUX3={ch[ARM_CH]:4d} AUX2={ch[AUX2_CH]:4d} AUX4={mu:4d}({fmode}) {rec_txt} | "
        f"{vic_txt} FC:{fm} {v} rx={bytes_rx}{thr_hint}"
    )
    sys.stdout.flush()


def run(args):
    cal = load_cal(args.cal_file)
    js = Joystick(args.js)
    print(f"Joystick: {js.name}  ({args.js})")
    print(f"Mapping:  roll=axis[{cal['axes']['roll']['axis']}] "
          f"pitch=axis[{cal['axes']['pitch']['axis']}] "
          f"thr=axis[{cal['axes']['throttle']['axis']}] "
          f"yaw=axis[{cal['axes']['yaw']['axis']}] "
          f"arm={cal['arm']['kind']}[{cal['arm']['index']}]")

    # Recording: the blackbox switch starts THREE synchronized streams at the
    # flick — FC blackbox (AUX2, relayed), laptop video, laptop vicon — and all
    # stop at disarm. Each flight gets its own session folder.
    rec = cal.get("record")
    recorder = vicon = cmd_log = telem = None
    vicon_off_reason = None
    if rec is None:
        print("Recording: no 'record' switch in calibration — AUX2 held off, "
              "no video/vicon.")
    else:
        # Command + telemetry loggers run whenever there's a session (no thread,
        # no hot-path I/O — they buffer in RAM and flush at end_session).
        cmd_log = CommandLogger()
        telem = TelemetryLogger()
        if CV2_OK:
            recorder = VideoRecorder(config.DEVICE_INDEX, config.WIDTH, config.HEIGHT,
                                     out_height=getattr(config, "VIDEO_OUT_HEIGHT", None))
            video_msg = f"video /dev/video{config.DEVICE_INDEX}"
        else:
            video_msg = "video OFF (no cv2 — run with .venv/bin/python)"
        if VICON_OK and not args.dry_run:
            vicon = ViconRecorder()
            # Connect + run the original startup sample-rate determination ONCE
            # now (fail-soft: if Vicon isn't streaming this disables it and we
            # fly without it). Keeping it here — not per session — means each
            # recording starts instantly at the switch flick, synced to t0.
            print(f"Vicon:    probing UDP :{vicon.port} for a stream "
                  f"(up to {_VICON_PROBE_TIMEOUT_S:.0f}s)…")
            if not vicon.prepare():
                vicon_off_reason = vicon.status
                vicon_msg = f"{CSI}1;31mvicon OFF ({vicon.status}){CSI}0m"
                vicon = None
            else:
                vicon_msg = f"{CSI}32mvicon {vicon.status}{CSI}0m"
        elif VICON_OK and args.dry_run:
            vicon_msg = "vicon OFF (dry-run)"
        else:
            vicon_off_reason = f"deps missing ({_VICON_ERR})"
            vicon_msg = "vicon OFF (deps missing)"
        print(f"Recording: blackbox switch {rec['kind']}[{rec['index']}] → "
              f"AUX2 + {video_msg} + {vicon_msg}")
        print(f"           + commands.csv / telemetry.csv (+ raw) every session")
        print(f"           sessions → {REC_DIR}/<timestamp>/  "
              f"(records while ARMED + switch ON)")
        # Loud, unmissable warning if Vicon won't record this flight (not dry-run).
        if vicon is None and not args.dry_run:
            _warn_vicon_off(vicon_off_reason)

    ser = None
    ranger_port = None
    if not args.dry_run:
        ranger_port = args.port or autodetect_port()
        if not ranger_port:
            js.close()
            sys.exit("No Ranger serial port found. Plug in the Ranger USB-C, or "
                     "pass the port explicitly: joystick_flight.py /dev/ttyACM0")
        ser = open_ranger(ranger_port, args.baud)
        print(f"Ranger:   {ranger_port} @ {args.baud} baud")
    else:
        print("Ranger:   (dry-run — no serial opened, nothing transmitted)")

    print("\nArm is edge-gated: flip the ARM switch to DISARMED once to enable "
          "arming.\nCtrl-C to stop (sends disarm).\n")

    parser = CrsfParser()
    # MSP_RAW_IMU (accel+gyro) is NOT a passive CRSF telemetry frame — we must
    # actively poll the FC for it (MSP-over-CRSF) and reassemble the chunked
    # response. Same proven path as common/live_telemetry.py. Polled at 10 Hz; the
    # request rides the uplink after the RC frame so it never delays the sticks.
    crsf_msp_parser = CrsfMspParser()
    msp_imu_req = build_crsf_msp_v2_request(MSP_RAW_IMU_CMD)
    flight_mode = None
    pack_v = None
    bytes_rx = 0
    seen_disarmed = False
    prev_want_record = False

    period = 1.0 / config.TX_HZ
    nxt = time.monotonic()
    last_ping = 0.0
    last_msp = 0.0
    last_render = 0.0
    PREVIEW_WIN = "drone feed — recording status"
    window_open = False

    def send_disarm():
        if ser is None:
            return
        ch = [config.NEUTRAL_US] * 16
        ch[config.CH_THR] = config.IDLE_THR_US
        ch[ARM_CH] = ARM_DISARMED_US
        ch[AUX2_CH] = AUX2_MID_US   # MID = nothing; never send LOW (=erase) here
        frame = build_rc_channels_packed(ch)
        for _ in range(5):
            ser.write(frame)
            time.sleep(0.01)

    # A "session" = one switch-on→disarm window. begin/end start & stop the
    # laptop recorders together off a single shared wall-clock t0 (the FC
    # blackbox starts on its own when it sees AUX2 go high, ~one frame later).
    session = {"dir": None, "t0": None, "stamp": None}

    def begin_session():
        t0 = time.time()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sdir = os.path.join(REC_DIR, stamp)
        os.makedirs(sdir, exist_ok=True)
        # Per-session blackbox/ subfolder — drop this flight's downloaded .bbl
        # here after landing; combine_flight.py looks here first.
        os.makedirs(os.path.join(sdir, "blackbox"), exist_ok=True)
        session.update(dir=sdir, t0=t0, stamp=stamp)
        if recorder is not None:
            recorder.start(t0, os.path.join(sdir, "video.mkv"))
        if vicon is not None:
            vicon.start(t0, os.path.join(sdir, "vicon.mat"))
        if cmd_log is not None:
            cmd_log.start(t0, os.path.join(sdir, "commands.csv"),
                          js.n_axes, js.n_buttons)
        if telem is not None:
            telem.start(t0, os.path.join(sdir, "telemetry.csv"),
                        os.path.join(sdir, "telemetry_raw.csv"))
        sys.stdout.write(f"\n{CSI}32m● REC SESSION {stamp}{CSI}0m → {sdir}\n")

    def end_session():
        if recorder is not None:
            recorder.stop()
        if vicon is not None:
            vicon.stop()
        if cmd_log is not None:
            cmd_log.save()
        if telem is not None:
            telem.save()
        if session["dir"]:
            _write_session_json(session, recorder, vicon, cmd_log, telem,
                                cal, ranger_port, args.baud, vicon_off_reason)
            sys.stdout.write(f"\n{CSI}33m■ SESSION SAVED{CSI}0m {session['stamp']}\n")
            if cmd_log is not None:
                sys.stdout.write(f"   commands: {len(cmd_log.rows)} frames\n")
            if telem is not None:
                sys.stdout.write(f"   telemetry: {len(telem.typed)} frames "
                                 f"({len(telem.raw)} raw)\n")
            if recorder is not None:
                sys.stdout.write(f"   video: {recorder.status}\n")
            if vicon is not None:
                sys.stdout.write(f"   vicon: {vicon.status}\n")
            sys.stdout.write(f"   → drop the FC .bbl into {session['dir']}/blackbox/ "
                             f"then run combine_flight.py\n")
        session.update(dir=None, t0=None, stamp=None)

    try:
        while True:
            js.poll()
            if not js.alive:
                print("\n!! joystick disconnected — disarming.", flush=True)
                send_disarm()
                break

            # Edge-gate: only enable arm passthrough after seeing DISARMED once.
            raw_armed = arm_is_armed(js, cal["arm"])
            if not raw_armed:
                seen_disarmed = True

            ch, armed, record_on = build_channels(js, cal, allow_arm=seen_disarmed)

            # Recording mirrors the drone's blackbox: capture while ARMED and the
            # blackbox switch is ON; stop + save when either drops (disarm or
            # switch off). Edge-triggered on the desired state so a failed start
            # isn't retried every loop.
            want_record = armed and record_on
            if want_record and not prev_want_record:
                begin_session()
            elif prev_want_record and not want_record:
                end_session()
            prev_want_record = want_record

            # Log every outgoing frame (all 16 channels + the full joystick
            # snapshot) while a session is active. RAM buffer only — no I/O here.
            if cmd_log is not None and session["dir"] is not None:
                cmd_log.log(time.time(), ch, armed, record_on, js.axes, js.buttons)

            if ser is not None:
                # Drain telemetry (non-blocking). Decode the known frames AND log
                # every frame raw (lossless), each stamped at laptop receive time.
                waiting = ser.in_waiting
                if waiting:
                    chunk = ser.read(waiting)
                    bytes_rx += len(chunk)
                    sess_active = telem is not None and session["dir"] is not None
                    for ftype, payload in parser.feed(chunk):
                        t_wall = time.time()
                        if sess_active:
                            telem.log_raw(t_wall, ftype, payload)
                        if ftype == T_FLIGHT_MODE:
                            flight_mode = decode_flight_mode(payload)
                            if sess_active:
                                telem.log_typed(t_wall, "flight_mode",
                                                {"flight_mode": flight_mode})
                        elif ftype == T_BATTERY:
                            b = decode_battery(payload)
                            if b:
                                pack_v = b["voltage_V"]
                                if sess_active:
                                    telem.log_typed(t_wall, "battery", {
                                        "bat_v": b["voltage_V"], "bat_a": b["current_A"],
                                        "bat_mah": b["capacity_mAh"],
                                        "bat_pct": b["remaining_pct"]})
                        elif ftype == T_ATTITUDE:
                            a = decode_attitude(payload)
                            if a and sess_active:
                                telem.log_typed(t_wall, "attitude", {
                                    "att_pitch_deg": a["pitch_deg"],
                                    "att_roll_deg": a["roll_deg"],
                                    "att_yaw_deg": a["yaw_deg"]})
                        elif ftype == T_LINK_STATS:
                            lk = decode_link_stats(payload)
                            if lk and sess_active:
                                telem.log_typed(t_wall, "link", {
                                    "up_lq": lk["up_lq"], "dn_lq": lk["dn_lq"],
                                    "up_rssi_dbm": lk["up_rssi1_dBm"],
                                    "dn_rssi_dbm": lk["dn_rssi_dBm"],
                                    "up_snr_db": lk["up_snr_dB"],
                                    "dn_snr_db": lk["dn_snr_dB"],
                                    "rf_mode": lk["rf_mode"],
                                    "active_ant": lk["active_ant"],
                                    "up_tx_pwr_idx": lk["up_tx_pwr_idx"]})
                        elif ftype == T_DEVICE_INFO and sess_active:
                            if len(payload) >= 2:
                                src = payload[1]
                                name = payload[2:].split(b"\x00", 1)[0].decode(
                                    "ascii", errors="replace")
                                if name:
                                    telem.log_typed(t_wall, "device_info", {
                                        "device_addr": f"0x{src:02X}",
                                        "device_name": name})
                        elif ftype == T_MSP_RESP:
                            # Reassemble chunked MSP-over-CRSF; log accel+gyro
                            # (converted to g / deg-per-s) when a full frame lands.
                            for mcmd, mpl in crsf_msp_parser.feed_chunk(payload):
                                if mcmd != MSP_RAW_IMU_CMD:
                                    continue
                                raw = decode_msp_raw_imu(mpl)
                                if raw and sess_active:
                                    c = convert_msp_raw_imu_units(raw)
                                    telem.log_typed(t_wall, "imu", {
                                        "imu_ax_g": c["ax_g"], "imu_ay_g": c["ay_g"],
                                        "imu_az_g": c["az_g"], "imu_gx_dps": c["gx_dps"],
                                        "imu_gy_dps": c["gy_dps"], "imu_gz_dps": c["gz_dps"],
                                        "imu_mag": c["mag_norm"]})
                # Send RC frame.
                try:
                    ser.write(build_rc_channels_packed(ch))
                except Exception as e:
                    print(f"\n!! Ranger write failed: {e} — stopping.", flush=True)
                    break
                now = time.monotonic()
                if now - last_ping > 2.0:
                    ser.write(build_device_ping())
                    last_ping = now
                if now - last_msp > 0.1:        # poll MSP_RAW_IMU at 10 Hz
                    ser.write(msp_imu_req)
                    last_msp = now

            now = time.monotonic()
            if now - last_render > 0.066:   # ~15 Hz
                render(ch, armed, seen_disarmed, raw_armed, record_on, recorder,
                       vicon, flight_mode, pack_v, bytes_rx, args.dry_run)
                # Live preview window: the feed + recording status while a session
                # is recording; closed between sessions. cv2 GUI must run on the
                # main thread (capture stays in the recorder thread), so drive it
                # here, throttled with the text render so it can't slow the RC loop.
                if recorder is not None and CV2_OK:
                    if recorder.recording:
                        frame = recorder.get_latest_frame()
                        if frame is not None:
                            disp = frame.copy()
                            vic = vicon.samples if (vicon and vicon.recording) else 0
                            cv2.putText(disp, f"REC  {recorder.frames}f   vicon {vic}",
                                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 0, 255), 2)
                            cv2.putText(disp, "ARMED" if armed else "DISARMED",
                                        (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 200, 0) if armed else (0, 200, 255), 2)
                            cv2.imshow(PREVIEW_WIN, disp)
                            cv2.waitKey(1)
                            window_open = True
                    elif window_open:
                        cv2.destroyAllWindows()
                        window_open = False
                last_render = now

            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        print("\nCtrl-C — disarming.", flush=True)
        send_disarm()
    finally:
        # If we exit mid-session (Ctrl-C / disconnect while recording), stop and
        # save both streams so nothing is lost.
        if session["dir"]:
            end_session()
        if window_open and CV2_OK:
            cv2.destroyAllWindows()
        js.close()
        if ser is not None:
            send_disarm()
            ser.close()
        print("Stopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", default=None,
                    help="Ranger serial port (default: autodetect)")
    ap.add_argument("baud", nargs="?", type=int, default=420000,
                    help="Ranger baud (default: 420000)")
    ap.add_argument("--js", default="/dev/input/js0", help="joystick device")
    ap.add_argument("--cal-file", default=CAL_FILE_DEFAULT,
                    help="calibration JSON path")
    ap.add_argument("--calibrate", action="store_true",
                    help="run full interactive calibration and exit")
    ap.add_argument("--calibrate-mode", action="store_true",
                    help="capture ONLY the 3-position flight-mode switch (AUX4: "
                         "low=acro/mid=angle/high=horizon) and merge into the "
                         "existing calibration; exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="read joystick + print sticks; never open Ranger / never send")
    args = ap.parse_args()

    if args.calibrate or args.calibrate_mode:
        js = Joystick(args.js)
        try:
            if args.calibrate:
                calibrate(js, args.cal_file)
            else:
                calibrate_mode_only(js, args.cal_file)
        finally:
            js.close()
        return
    run(args)


if __name__ == "__main__":
    main()
