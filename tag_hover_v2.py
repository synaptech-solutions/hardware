"""AprilTag hover controller v2 — cascaded position controller, FC in Angle mode.

Architecture
------------
  Outer X-position PID → desired roll angle (deg)
  Outer Z-position PID → desired pitch angle (deg)
  Altitude velocity-loop PD → throttle (us)
  Yaw P+deadband → yaw (us)

Desired roll/pitch angles are converted to CRSF us via the FC's Angle-mode
stick scaling; the FC's attitude PID closes the inner loop.

State machine (matches tag_hover_controller.py):
  DRYRUN     — print sticks, never arm. Set config.DRY_RUN = True.
  ARM_SETTLE — wait for FC to clear ARMING_DISABLED + a stable vision lock,
               then capture target_fwd / target_lat / yaw_at_arm and enter FLY.
  FLY        — closed-loop hover until tag-lost timeout or kill key.

Run:  .venv/bin/python tag_hover_v2.py
Toggle DRY_RUN in controller_v2/config.py before flying.
"""
import os
import sys
import time
import math
import threading
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from controller_v2 import config
from controller_v2.threads import (
    Shared, TxState, vision_loop, tx_loop, rx_loop,
    open_ranger, autodetect_port,
)
from controller_v2.pose_filter import PoseFilter
from controller_v2.cascaded import CascadedHoverController


CALIB_FILE = os.path.join(HERE, "camera_setup", "camera_calibration.npz")
LOG_DIR = os.path.join(HERE, "flight_logs")
LOG_FILE = os.path.join(LOG_DIR, "hover_controller_v2.log")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        sys.exit(f"no {CALIB_FILE} — run camera_setup/calibrate_camera.py first")
    c = np.load(CALIB_FILE)
    K = c["K"].astype(np.float64)
    dist = c["dist"].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"calibration loaded (RMS={float(c['rms']):.3f}px)")
    return K, dist, (fx, fy, cx, cy)


def is_armed(mode):
    return mode is not None and not mode.endswith("*") and not mode.startswith("!")


def axis_cmd_label(us, pos_name, neg_name):
    d = int(us - config.NEUTRAL_US)
    mag = abs(d)
    if mag == 0:
        return f"{pos_name}/{neg_name} HOLD [0]"
    if d > 0:
        return f"{pos_name} [{mag}]"
    return f"{neg_name} [{mag}]"


def throttle_cmd_label(us, deadband_us=5):
    d = int(us - config.HOVER_THROTTLE_US)
    mag = abs(d)
    if mag <= deadband_us:
        return f"THROTTLE HOLD [{us}]"
    if d > 0:
        return f"THROTTLE UP [{mag}] (raw {us})"
    return f"THROTTLE DOWN [{mag}] (raw {us})"


def main():
    K, dist, cam_params = load_calibration()
    port = autodetect_port()
    if not port:
        sys.exit("Ranger not found on USB. Plug it in and retry.")
    ser = open_ranger(port, 420000)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_fp = open(LOG_FILE, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")

    log(f"===== RUN START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log(f"CRSF on {port} @ 420000")
    log(f"controller_v2 cascaded (X/Z pos PIDs → angle → us, FC Angle mode)")
    log(f"angle scaling: {config.STICK_US_PER_DEG:.3f} us/deg "
        f"(angle_limit={config.FC_ANGLE_LIMIT_DEG:.1f}°) "
        f"— TODO: verify against Air75 CLI `get angle_limit`")

    shared = Shared()
    tx = TxState()
    threading.Thread(target=vision_loop,
                     args=(shared, tx, K, dist, cam_params),
                     daemon=True).start()
    threading.Thread(target=tx_loop, args=(ser, tx), daemon=True).start()
    threading.Thread(target=rx_loop, args=(ser, tx), daemon=True).start()

    cv2.namedWindow("hover_v2", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("hover_v2", 1280, 720)

    pose_filter = PoseFilter()
    controller = CascadedHoverController()
    armed_cmd = False
    target_fwd_m = config.TARGET_FWD_FALLBACK_M
    target_lat_m = config.TARGET_LAT_M
    target_up_m = config.TARGET_UP_M
    yaw_at_arm = 0.0
    arm_lock_since = None
    last_thr = float(config.IDLE_THR_US)
    last_state_t = time.monotonic()
    tx_period = 1.0 / config.TX_HZ
    next_loop = time.monotonic()
    last_display_t = 0.0
    last_log_t = 0.0
    dry_targets_set = False     # DRYRUN: lock targets + yaw_at_arm on first vision lock

    def disarm_and_exit(reason):
        log(f"DISARM: {reason}")
        dis = [config.NEUTRAL_US] * 16
        dis[config.CH_THR] = config.IDLE_THR_US
        dis[config.CH_ARM] = config.ARM_LOW_US
        dis[config.CH_PREARM] = config.PREARM_LOW_US
        dis[config.CH_MODE] = config.MODE_ANGLE_US
        tx.set(dis)
        time.sleep(0.25)
        tx.stop = True
        shared.running = False
        time.sleep(0.05)
        ser.close()
        cv2.destroyAllWindows()
        if not log_fp.closed:
            log_fp.write(f"===== RUN END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_fp.close()

    if config.DRY_RUN:
        log("*** DRY RUN — will NOT arm ***")
        state = "DRYRUN"
    else:
        log("*** LIVE FLIGHT ***")
        ans = input("Area clear? Hand on battery? Type FLY to arm: ").strip()
        log(f"pilot confirmation input={ans!r}")
        if ans != "FLY":
            disarm_and_exit("not confirmed by user")
            return
        log("waiting for link + telemetry...")
        t0 = time.monotonic()
        last_diag = 0.0
        while tx.flight_mode is None and time.monotonic() - t0 < 8.0:
            now = time.monotonic()
            if now - last_diag >= 1.0:
                log(f"  waiting... rx_bytes={tx.bytes_rx} rx_frames={tx.frames_rx} "
                    f"att={tx.att_stamp > 0} batt={tx.pack_v}")
                last_diag = now
            time.sleep(0.1)
        if tx.flight_mode is None:
            disarm_and_exit(f"no telemetry (rx_bytes={tx.bytes_rx} "
                            f"rx_frames={tx.frames_rx}) — FC busy or link down")
            return
        log(f"link up, mode={tx.flight_mode!r}")
        if tx.flight_mode.startswith("!"):
            disarm_and_exit(f"FC arm-blocked: {tx.flight_mode!r}")
            return
        state = "ARM_SETTLE"
    last_state_t = time.monotonic()

    prev_loop_t = time.monotonic()
    try:
        while shared.running:
            now = time.monotonic()
            dt = clamp(now - prev_loop_t, 0.0, config.ESTIMATOR_MAX_DT_S)
            prev_loop_t = now

            flight_mode = tx.flight_mode
            pack_v = tx.pack_v
            if (config.LOW_BATT_CUTOFF and pack_v is not None
                    and config.BATT_PRESENT_V < pack_v
                    < config.MIN_CELL_V * config.CELLS):
                disarm_and_exit(f"battery low ({pack_v:.2f}V)")
                break
            if armed_cmd and flight_mode and flight_mode.startswith("!"):
                disarm_and_exit(f"FC refused/err mode {flight_mode!r}")
                break

            # Pull latest vision pose.
            with shared.lock:
                vis_pos = shared.pos
                vis_stamp = shared.pos_stamp
                vis_dm = shared.decision_margin
                vis_has = shared.has_tag
                vis_ids = shared.ids_seen
                vis_det = shared.detect_label
                vis_dbg = shared.vision_dbg
                frame = shared.frame
            vis_age = now - vis_stamp if vis_stamp else 1e9
            vis_fresh = vis_has and vis_age < config.VISION_FRESH_S

            # Pose filter update.
            if state in ("DRYRUN", "ARM_SETTLE"):
                if vis_fresh:
                    pose_filter.reset(vis_pos, now)
                pose_filter.update(vis_pos, vis_fresh, now, ground_pinned=True)
            else:
                ground_pinned = last_thr < (config.HOVER_THROTTLE_US
                                            - config.GROUND_PIN_BAND_US)
                pose_filter.update(vis_pos, vis_fresh, now,
                                   ground_pinned=ground_pinned)

            est_fwd, est_lat, est_up = pose_filter.p
            v_fwd, v_lat, v_up = pose_filter.v
            est_age = pose_filter.age(now) if state == "FLY" else 0.0

            state_dict = {
                "est_fwd": est_fwd, "est_lat": est_lat, "est_up": est_up,
                "v_fwd": v_fwd, "v_lat": v_lat, "v_up": v_up,
                "yaw_deg": tx.yaw_deg,
            }

            roll = pitch = yaw = config.NEUTRAL_US
            thr = config.IDLE_THR_US
            arm = config.ARM_LOW_US
            prearm = config.PREARM_LOW_US
            note = ""
            ctl_out = None
            lost = 0.0

            if state == "DRYRUN":
                if pose_filter.inited:
                    if not dry_targets_set:
                        # DRYRUN aims to CENTER the tag in the camera: lat=0
                        # (and yaw bearing→0, handled in the yaw loop), holding
                        # the nominal forward distance. This is the only
                        # difference from FLIGHT, which instead locks fwd/lat to
                        # whatever the drone reads at launch (see ARM_SETTLE).
                        controller.set_targets(
                            target_fwd_m=config.TARGET_FWD_FALLBACK_M,
                            target_lat_m=0.0,
                            target_up_m=target_up_m,
                            yaw_at_arm_deg=tx.yaw_deg,
                        )
                        dry_targets_set = True
                        log(f"DRY targets locked (center tag): "
                            f"fwd={config.TARGET_FWD_FALLBACK_M:+.2f} lat=+0.00 "
                            f"up={target_up_m:+.2f} yaw_ref={tx.yaw_deg:+.1f}")
                    ctl_out = controller.step(state_dict, dt)
                    roll = ctl_out["roll_us"]
                    pitch = ctl_out["pitch_us"]
                    yaw = ctl_out["yaw_us"]
                    thr = ctl_out["throttle_us"]
                    note = (f"DRY est=({est_fwd:+.2f},{est_lat:+.2f},{est_up:+.2f}) "
                            f"v=({v_fwd:+.2f},{v_lat:+.2f},{v_up:+.2f}) "
                            f"des(R,P)=({ctl_out['desired_roll_deg']:+.2f},"
                            f"{ctl_out['desired_pitch_deg']:+.2f})deg")
                else:
                    note = "DRY waiting for vision"

            elif state == "ARM_SETTLE":
                prearm = config.PREARM_HIGH_US
                if now - last_state_t < 0.4:
                    arm = config.ARM_LOW_US
                    note = "prearm…"
                else:
                    arm = config.ARM_HIGH_US
                    armed_cmd = True
                    if flight_mode and flight_mode.startswith("!"):
                        disarm_and_exit(f"won't arm: {flight_mode!r}")
                        break
                    if is_armed(flight_mode):
                        strong = (vis_fresh
                                  and vis_dm >= config.MIN_TAG_DECISION_MARGIN)
                        if strong:
                            arm_lock_since = arm_lock_since or now
                            lock_s = now - arm_lock_since
                            if lock_s < config.ARM_LOCK_MIN_S:
                                note = (f"lock {lock_s:.2f}/"
                                        f"{config.ARM_LOCK_MIN_S:.2f}s "
                                        f"dm={vis_dm:.0f}")
                            else:
                                target_fwd_m = (
                                    float(vis_pos[0]) if config.LOCK_FWD_AT_ARM
                                    else config.TARGET_FWD_FALLBACK_M)
                                target_lat_m = (
                                    float(vis_pos[1]) if config.LOCK_LAT_AT_ARM
                                    else config.TARGET_LAT_M)
                                yaw_at_arm = tx.yaw_deg
                                pose_filter.reset(vis_pos, now)
                                last_thr = float(config.IDLE_THR_US)
                                controller.set_targets(
                                    target_fwd_m=target_fwd_m,
                                    target_lat_m=target_lat_m,
                                    target_up_m=target_up_m,
                                    yaw_at_arm_deg=yaw_at_arm,
                                )
                                state, last_state_t = "FLY", now
                                log(f">> FLY: target fwd={target_fwd_m:+.2f} "
                                    f"lat={target_lat_m:+.2f} up=0  "
                                    f"yaw_arm={yaw_at_arm:+.1f}")
                        else:
                            arm_lock_since = None
                            note = (f"waiting for vision lock "
                                    f"vis_age={vis_age*1000:.0f}ms "
                                    f"dm={vis_dm:.0f}")
                    elif now - last_state_t > config.ARM_CONFIRM_S:
                        disarm_and_exit(
                            f"arm not confirmed after "
                            f"{config.ARM_CONFIRM_S:.1f}s (mode={flight_mode!r}). "
                            f"FC may still be doing gyro cal — wait a few "
                            f"seconds after powering the drone before launching.")
                        break
                    else:
                        note = (f"FC arm-blocked (mode={flight_mode!r}) "
                                f"t={now-last_state_t:.1f}/"
                                f"{config.ARM_CONFIRM_S:.1f}s "
                                f"att_stamp={tx.att_stamp > 0}")

            elif state == "FLY":
                arm = config.ARM_HIGH_US
                prearm = config.PREARM_HIGH_US
                if est_age < config.STATE_MAX_AGE_S:
                    ctl_out = controller.step(state_dict, dt)
                    roll = ctl_out["roll_us"]
                    pitch = ctl_out["pitch_us"]
                    yaw = ctl_out["yaw_us"]
                    thr = ctl_out["throttle_us"]
                    last_thr = thr
                    note = (f"fly fwd={est_fwd:+.2f}(t{target_fwd_m:+.2f}) "
                            f"lat={est_lat:+.2f}(t{target_lat_m:+.2f}) "
                            f"up={est_up:+.2f}  "
                            f"des(R,P)=({ctl_out['desired_roll_deg']:+.2f},"
                            f"{ctl_out['desired_pitch_deg']:+.2f})deg  "
                            f"v=({v_fwd:+.2f},{v_lat:+.2f},{v_up:+.2f})")
                else:
                    lost = est_age - config.STATE_MAX_AGE_S
                    if est_age > config.KILL_AFTER_LOST_S:
                        disarm_and_exit(f"tag lost {est_age:.1f}s")
                        break
                    bleed = config.LAND_DESCENT_US_PER_S * lost
                    thr = round(clamp(config.HOVER_THROTTLE_US - bleed,
                                      config.IDLE_THR_US,
                                      config.HOVER_THROTTLE_US))
                    last_thr = thr
                    controller.reset()         # don't let integrators run while blind
                    note = f"LOST {est_age:.2f}s — descending thr={thr}"

            ch = [config.NEUTRAL_US] * 16
            ch[config.CH_ROLL] = int(roll)
            ch[config.CH_PITCH] = int(pitch)
            ch[config.CH_YAW] = int(yaw)
            ch[config.CH_THR] = int(clamp(thr, 1000, 2000))
            ch[config.CH_ARM] = arm
            ch[config.CH_MODE] = config.MODE_ANGLE_US
            ch[config.CH_PREARM] = prearm
            tx.set(ch)

            if frame is not None:
                cv2.putText(frame,
                            f"[{state}] R{ch[config.CH_ROLL]} "
                            f"P{ch[config.CH_PITCH]} "
                            f"Y{ch[config.CH_YAW]} T{ch[config.CH_THR]} "
                            f"{'ARM' if arm > 1500 else 'safe'}",
                            (10, config.HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panel_lines = [
                    axis_cmd_label(ch[config.CH_ROLL], "ROLL RIGHT", "ROLL LEFT"),
                    axis_cmd_label(ch[config.CH_PITCH], "PITCH UP", "PITCH DOWN"),
                    axis_cmd_label(ch[config.CH_YAW], "YAW RIGHT", "YAW LEFT"),
                    throttle_cmd_label(ch[config.CH_THR]),
                ]
                if ctl_out is not None:
                    panel_lines.append(
                        f"des roll={ctl_out['desired_roll_deg']:+.1f}° "
                        f"pitch={ctl_out['desired_pitch_deg']:+.1f}°"
                    )
                y0 = 58
                for line in panel_lines:
                    cv2.putText(frame, line, (10, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 255, 255), 2)
                    y0 += 24
                if now - last_display_t >= (1.0 / max(config.DISPLAY_HZ, 1.0)):
                    cv2.imshow("hover_v2", frame)
                    last_display_t = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27, ord(' ')):
                disarm_and_exit("kill key")
                break

            log_due = (config.LOG_EVERY_LOOP
                       or (now - last_log_t >= (1.0 / max(config.LOG_HZ, 1.0))))
            if note and log_due:
                mode_txt = flight_mode if flight_mode is not None else "n/a"
                batt_txt = f"{pack_v:.2f}V" if pack_v is not None else "n/a"
                vis_txt = (f"vis={'fresh' if vis_fresh else 'lost'} "
                           f"age={vis_age*1000:.0f}ms dm={vis_dm:.0f} "
                           f"det={vis_det or 'none'} ids={vis_ids or '-'}")
                imu_txt = (f"imu(r/p/y)=({tx.roll_deg:+.1f},"
                           f"{tx.pitch_deg:+.1f},{tx.yaw_deg:+.1f})deg")
                if ctl_out is not None:
                    ctl_txt = (f"e=(f{ctl_out['e_fwd_b']:+.3f},"
                               f"l{ctl_out['e_lat_b']:+.3f},"
                               f"u{ctl_out['e_up']:+.3f}) "
                               f"e_yaw={math.degrees(ctl_out['e_yaw_rad']):+.1f}deg "
                               f"des(R,P)=({ctl_out['desired_roll_deg']:+.2f},"
                               f"{ctl_out['desired_pitch_deg']:+.2f})deg "
                               f"intI(X,Z)=({controller.pid_x.integral:+.3f},"
                               f"{controller.pid_z.integral:+.3f})")
                else:
                    ctl_txt = "ctl=n/a"
                log(f"[{state}] {note}  -> R{ch[config.CH_ROLL]} "
                    f"P{ch[config.CH_PITCH]} Y{ch[config.CH_YAW]} "
                    f"T{ch[config.CH_THR]}  {vis_txt}  {imu_txt}  {ctl_txt}  "
                    f"est_age={est_age*1000:.0f}ms dt={dt*1000:.1f}ms "
                    f"lost={lost:.3f}  vdbg=[{vis_dbg}] "
                    f"vis=[{shared.vision_status}] mode={mode_txt} "
                    f"batt={batt_txt}")
                last_log_t = now

            next_loop += tx_period
            sleep = next_loop - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_loop = time.monotonic()
    except KeyboardInterrupt:
        disarm_and_exit("Ctrl-C")
    finally:
        if shared.running:
            disarm_and_exit("loop exit")
        elif not log_fp.closed:
            log_fp.write(f"===== RUN END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_fp.close()


if __name__ == "__main__":
    main()
