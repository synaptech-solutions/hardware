# Synetic Labs — Drone Hardware & Flight Data Collection

Tooling for flying a small FPV drone by hand (or autonomously) while recording
**everything that flows through the laptop**, then merging it with the flight
controller's blackbox and motion-capture pose into a single time-aligned file for
analysis (latency, control response, trajectory, etc.).

There are **two data-collection pipelines** and a separate autonomous controller:

| Pipeline | Folder | Trigger | Use it when |
|---|---|---|---|
| **A — TX12 synced collection** | [`data_logging/`](data_logging/) | blackbox switch on the radio | You fly by hand through the laptop (TX12 → Ranger → drone) and want commands + telemetry + video + Vicon + FC blackbox, deterministically aligned. **This is the main pipeline.** |
| **B — Standalone Vicon+video recorder** | [`pycode_ViCON/`](pycode_ViCON/) | SPACE key | Any drone (no TX12 relay) — record Vicon pose + video, then post-sync with the blackbox via yaw cross-correlation. |
| Autonomous hover controllers | [`drone_control/`](drone_control/) | TX12 arm + record | Closed-loop hover (flight control). **`Vicon_control/`** — live-Vicon hover (current); **`apriltag_control/`** — AprilTag hover (archived). Records via the same pipeline as A. |

---

## 1. Hardware

- **Drone:** BetaFPV **Air75** — analog whoop, Betaflight FC, **no GPS**. Flight modes on **AUX4**: acro / angle / horizon.
- **RF link:** ELRS **"Ranger" TX module** over **USB-C**, exposing full bidirectional **CRSF at 420000 baud** (custom baud set via a `TCSETS2` ioctl — plain `pyserial` at 420000 fails with `EINVAL`).
- **Radio:** RadioMaster **TX12** (EdgeTX) in **USB Joystick (HID)** mode → appears as `/dev/input/js0` (7 axes, 24 buttons). Sticks/switches are read from the joystick and re-emitted as CRSF out the Ranger.
- **Video:** drone analog feed → analog VRX → HDMI → **Cam Link 4K** → `/dev/video4`, captured at **1280×720 MJPG** (resolution must match `camera_setup/camera_calibration.npz`).
- **Motion capture:** **Vicon**, streaming rigid-body pose over **UDP :51001** (drone = rigid body **`b1`**), logged at ~100 Hz.

### CRSF channel map (TX12 pipeline)

| Channel | CRSF index | Control | Values (µs) |
|---|---|---|---|
| Roll / Pitch / Throttle / Yaw | 0 / 1 / 2 / 3 | gimbals (AETR) | 1000–2000 |
| AUX2 — blackbox switch | 5 | 3-pos | **HIGH 2000** = start FC blackbox **+ trigger laptop recording**; MID 1500 = nothing; **LOW 1000 = ERASE dataflash** |
| AUX3 — arm | 6 | 2-pos | armed 1508 / disarmed 1000 (edge-gated) |
| AUX4 — flight mode | 7 | 3-pos | LOW 1000 = **acro**, MID 1500 = **angle**, HIGH 1900 = **horizon** |

---

## 2. Environment

Everything runs on the repo virtualenv **`.venv`** (Python 3.12, uv-managed) which has
`numpy`, `scipy`, `opencv` (`cv2`), `matplotlib`, `dash`, `plotly`, `tqdm`.

```bash
.venv/bin/python <script>          # always use the venv python
```

The system `python3` works only for dependency-free bits (e.g. the joystick
monitor); video/Vicon/plots/dashboard need the venv.

The blackbox decoder must be built once:
`pycode_ViCON/tools/blackbox-tools/obj/blackbox_decode` (from Betaflight blackbox-tools).

---

## 3. Pipeline A — TX12 synced data collection (main)

You fly by hand: **TX12 (USB joystick) → laptop → Ranger (CRSF) → drone**. The laptop
relays your sticks *and* records every stream on one shared clock.

### Scripts (`data_logging/`)

| Script | Purpose |
|---|---|
| [`joystick_flight.py`](data_logging/joystick_flight.py) | Fly + record. The one you run to collect data. |
| [`monitor_tx12.py`](data_logging/monitor_tx12.py) | Live view of **every** joystick axis/button (diagnostics / finding a switch). |
| [`combine_flight.py`](data_logging/combine_flight.py) | Post-flight merge → `flight_synced.csv`. |
| [`plot_flight.py`](data_logging/plot_flight.py) | Static PNG overview of a synced flight. |
| [`dashboard_flight.py`](data_logging/dashboard_flight.py) | Interactive web dashboard of a synced flight. |

### 3a. One-time calibration

EdgeTX's channel→USB-axis assignment is radio-specific, so which axis is roll/pitch/etc.
is discovered empirically and saved to `tx12_joystick_cal.json`.

```bash
.venv/bin/python data_logging/joystick_flight.py --calibrate        # full: gimbals, arm, blackbox, mode
.venv/bin/python data_logging/joystick_flight.py --calibrate-mode    # just the flight-mode switch, merged in
```

Calibration only needs the TX12 connected (no Ranger/drone). Hold each control at the
named position and press Enter. For the **mode switch** the switch must first be mapped
to a **joystick axis** in EdgeTX (the radio's *USB Joystick* config exports channels 1–8
as 8 axes, 9–32 as buttons — but this radio's `/dev/input/js0` only enumerates 7 axes,
so a switch on CH8 is invisible; put it on a lower channel that maps to a visible axis).
Use the monitor to see what reaches the laptop:

```bash
python3 data_logging/monitor_tx12.py     # flip the switch, watch which axis lights up green
```

### 3b. Collecting a flight

```bash
.venv/bin/python data_logging/joystick_flight.py          # autodetect Ranger
.venv/bin/python data_logging/joystick_flight.py /dev/ttyUSB0 420000   # explicit port/baud
.venv/bin/python data_logging/joystick_flight.py --dry-run             # read sticks, never transmit
```

At launch it probes Vicon (up to 5 s). **If Vicon isn't streaming you get a big red
`VICON IS OFF` banner** — start Vicon Tracker (streaming to this laptop on :51001) and
restart before flying if you want pose.

Flow on the radio:
1. **Arm is edge-gated** — flip the arm switch to DISARMED once to enable arming (no arm-on-startup).
2. **Arm**, lower throttle (Betaflight refuses to arm with throttle > 1050 µs).
3. **Flip the blackbox switch HIGH** while armed → starts a **synchronized recording session** at one shared wall-clock `t0`: FC blackbox + laptop video + Vicon + commands + telemetry, all at once.
4. Fly. **Disarm** (or drop the blackbox switch) → the session stops and saves.
5. Ctrl-C to exit (sends explicit disarm frames).

**Live status line** (updates ~15 Hz) shows: stick µs (R/P/T/Y), arm state,
`AUX3`/`AUX2`/`AUX4(MODE)`, recording state, a persistent
**`VICON:OFF`(red)/`ready`(yellow)/`●N`(green)** indicator (frozen sample count = stream died),
and FC mode/voltage/bytes-received.

### 3c. Per-session folder

Each session is its own folder; a `blackbox/` subfolder is auto-created as your drop spot:

```
data_logging/recordings/<YYYYMMDD_HHMMSS>/
├── blackbox/<flight>.bbl     ← drop the FC blackbox here after landing
├── commands.csv              outgoing RC frames @TX_HZ (100 Hz): all 16 ch + full joystick
├── telemetry.csv             decoded incoming CRSF telemetry (typed, per frame)
├── telemetry_raw.csv         every incoming frame as hex (lossless)
├── video.mp4                 drone feed
├── video_frames.csv          real per-frame capture timestamps
├── vicon.mat                 Vicon pose @100 Hz (absent if Vicon was off)
├── session.json              t0, per-stream offsets, config snapshot, vicon status
├── flight_synced.csv         ← combine_flight.py output
└── flight_synced.meta.json   scalars / provenance / column list
```

> `data_logging/recordings/` is git-ignored, so all flight data (incl. `.bbl`, video) stays out of git.

**The shared clock:** every laptop stream is stamped `t_rel = wall − t0`, where `t0` is the
blackbox-switch flick. The FC blackbox runs on its own clock and is zero-based to its
first sample (≈ the same flick). So alignment needs no cross-correlation — just `--offset`
to nudge out the laptop→FC link latency (which you can *measure* from `commands` vs `rcCommand`).

#### What each capture file holds
- **`commands.csv`** — `t_rel, t_wall, ch00_us..ch15_us, armed, record_on, ax0..axN, btn0..btnM` (all 16 channels sent + the raw joystick before mapping).
- **`telemetry.csv`** — `t_rel, t_wall, type, …` one row per incoming CRSF frame; `type` ∈ {attitude, battery, link, imu, flight_mode, device_info} with that type's fields filled. **IMU (accel/gyro)** comes from actively polling MSP_RAW_IMU at 10 Hz (confirmed working on the Air75).
- **`telemetry_raw.csv`** — `t_rel, t_wall, frame_type, length, payload_hex` — every frame, re-decodable offline.
- **`video_frames.csv`** — `frame_idx, t_rel, t_wall` (real capture times, not assumed fps).
- **`vicon.mat`** — `Abs_time` (t_rel), `b1_x/y/z`, quaternion `b1_qx..qw`, velocities `b1_*_dot`.

### 3d. Merge → `flight_synced.csv`

Drop the flight's `.bbl` into the session's `blackbox/` subfolder, then:

```bash
.venv/bin/python data_logging/combine_flight.py                    # newest session + its .bbl
.venv/bin/python data_logging/combine_flight.py recordings/<stamp> # a specific session
.venv/bin/python data_logging/combine_flight.py <session> <log.bbl> --poles 12 --offset 0.015
```

It decodes the `.bbl`, resamples everything onto a master clock, and writes one wide CSV +
a `.meta.json` sidecar. **Master clock = Vicon `Abs_time` if present, else `commands.csv`**
(a Vicon-less flight still merges — it just has no pose).

Merged columns:
- `Abs_time` — master time base (s from t0)
- `b1_x/y/z`, `b1_qx..qw`, `b1_vx/vy/vz` — pose + velocity (only if Vicon recorded)
- `cmd_ch00_us..ch15_us`, `cmd_armed`, `cmd_record_on` — what we **sent**
- `tlm_*` — telemetry (attitude, link LQ/RSSI/SNR, battery, **live IMU** `tlm_imu_*`)
- `bb_*` — **every** decoded blackbox channel (gyro, accel, PID P/I/D/F, `rcCommand`, setpoint, motor, eRPM, debug, vbat…)
- `motor_rpm_0..3`, `motor_erpm_0..3`, `motor_cmd_0..3` — derived motor arrays

The headline analysis this unlocks: **`cmd_ch00_us` (sent) vs `bb_rcCommand_0` (FC received)** → uplink latency.

---

## 4. Pipeline B — Standalone Vicon + video recorder

For drones **not** flown through the TX12 relay. Recording is triggered by the **SPACE key**,
and the blackbox is merged afterward by cross-correlation (no shared trigger).

### Scripts (`pycode_ViCON/`)

| Script | Purpose |
|---|---|
| [`UdpReceiver_datacollection.py`](pycode_ViCON/UdpReceiver_datacollection.py) | `__main__` = SPACE-triggered **Vicon + video** recorder → `DataExchange/<stamp>.mat` + `.mp4`. (Also the shared Vicon-parsing library used by pipeline A.) |
| [`simple_video_recorder.py`](pycode_ViCON/simple_video_recorder.py) | Just the video feed + SPACE record toggle (no Vicon/blackbox). |
| [`sync_log.py`](pycode_ViCON/sync_log.py) | Merge a `.bbl` + a pose `.mat` → `*_synced.mat` by **yaw-rate cross-correlation** (do a sharp in-air yaw so both instruments see it). |
| [`plot_synced.py`](pycode_ViCON/plot_synced.py) | Static PNG of a `*_synced.mat`. |
| [`dashboard_synced.py`](pycode_ViCON/dashboard_synced.py) | Interactive web dashboard of a `*_synced.mat`. |

```bash
# 1. record (SPACE to start/stop; saves to DataExchange/)
.venv/bin/python pycode_ViCON/UdpReceiver_datacollection.py
# 2. merge with the blackbox
.venv/bin/python pycode_ViCON/sync_log.py LOG.bbl POSE.mat        # or no args = newest of each
# 3. view
.venv/bin/python pycode_ViCON/plot_synced.py                     # static PNG
.venv/bin/python pycode_ViCON/dashboard_synced.py                # web
```

Outputs live in `pycode_ViCON/DataExchange/`.

**A vs B:** pipeline A aligns on a deterministic shared trigger (the switch flick) and
captures the full laptop in/out streams; pipeline B has no shared trigger so it recovers
the offset by yaw cross-correlation and records only pose + video + blackbox.

---

## 5. Web dashboard

Both `dashboard_flight.py` (pipeline A, reads `flight_synced.csv`) and
`dashboard_synced.py` (pipeline B, reads `*_synced.mat`) are thin launchers over the
**shared module** [`pycode_ViCON/flight_dashboard.py`](pycode_ViCON/flight_dashboard.py) —
identical UI, **reads either CSV or .mat**.

```bash
.venv/bin/python data_logging/dashboard_flight.py          # newest flight
.venv/bin/python data_logging/dashboard_flight.py --pick   # choose from a menu
.venv/bin/python data_logging/dashboard_flight.py recordings/<stamp>/flight_synced.csv
.venv/bin/python data_logging/dashboard_flight.py --port 8060 --no-browser
```

Opens a Plotly Dash app at `http://127.0.0.1:8050`. Features:
- **2D panels grouped by data source** — *Vicon* (pose/vel/orientation), *Commands sent* (TRPY sticks, switches, flags), *Telemetry* (attitude, link, battery, IMU), *Blackbox* (gyro/accel/PID/rcCommand/setpoint/motors/…), *Derived*. Toggle any channel on/off.
- A master **time-window slider** drives both the 2D x-range and the 3D slice.
- A **3D trajectory** (color-by time / speed / altitude / RPM / throttle cmd …) with an optional orientation overlay.
- Per-graph fullscreen.

---

## 6. Autonomous controllers (`drone_control/`)

`drone_control/` is split into a shared layer + two controllers:

```
drone_control/
├── common/            shared hardware/IO layer (base dep for everything below + data_logging)
│   ├── channels.py     Air75 CRSF channel map + µs levels + camera (SINGLE SOURCE OF TRUTH)
│   ├── live_telemetry.py / ranger.py   CRSF/MSP build+parse, custom-baud Ranger open
│   ├── tx12.py / recorders.py          TX12 input + the synced recorders (extracted from joystick_flight)
│   └── pid.py
├── Vicon_control/     live-Vicon hover (current)
└── apriltag_control/  AprilTag hover (archived: tag_hover_v2.py, controller_v2/, camera_setup/, setup/)
```

### `Vicon_control/` — live-Vicon hover (current)

Closed-loop hover on **live Vicon pose** (100 Hz, absolute position + drift-free
yaw) — much easier than AprilTags (no dropouts, no dead-reckoning, no compass
hacks). The **TX12 stays in the loop for arm / disarm / record only**, and the
controller **records via the same pipeline as Pipeline A**, so its flights merge +
render identically.

- [`vicon_hover.py`](drone_control/Vicon_control/vicon_hover.py) — main loop. **Arm** on the TX12 (motors idle); flip the **record switch** to launch → the drone captures its takeoff pose, climbs `CLIMB_M` (default 1 m) and holds takeoff x/y + heading. Flip record back to land gently; **flick the arm switch to disarm = instant kill** at any time. FC stays in **Angle** mode (forced).
- `controller.py` — world→body position PID → desired roll/pitch angle; altitude PI velocity loop that *learns* hover throttle; P-on-absolute-heading yaw hold.
- `vicon_source.py` — owns the one Vicon receiver (shared with the recorder); gives world-frame `x,y,z,yaw,vx,vy,vz`.
- `config.py` — climb target, gains, hover band, safety limits. **Toggle `DRY_RUN`** (default `True`) before flying.
- `combine.py` / `plot.py` / `dashboard.py` — thin launchers over the Pipeline-A tools, defaulting to `flight_logs/`.

```bash
.venv/bin/python data_logging/joystick_flight.py --calibrate   # one-time TX12 cal (shared file)
.venv/bin/python drone_control/Vicon_control/vicon_hover.py --dry-run   # DRY_RUN: print control, never arm
.venv/bin/python drone_control/Vicon_control/vicon_hover.py             # fly (after dry-run dir-checks)
```

**Bring-up order (don't skip — props OFF until directions verified):**
1. `--dry-run`: arm + record on the TX12, then **move the drone by hand** off the hover point and confirm the printed `des(R,P)` / throttle push *toward* the target (this is the gate that catches a sign inversion).
2. Props off, on the bench: arm + record, confirm motors spin toward hover and fight a hand-induced displacement in the right direction.
3. First hover: set `CLIMB_M = 0.3` in `config.py`, fly, land, then raise to 1.0 m and tune gains.

### `apriltag_control/` — AprilTag hover (archived)

The earlier closed-loop **AprilTag** hover (never fully reliable — vision dropouts,
dead-reckoning pogo, compass-less yaw). Kept for reference.
- [`tag_hover_v2.py`](drone_control/apriltag_control/tag_hover_v2.py) — cascaded position PID → angles, altitude PI that learns hover. Toggle `DRY_RUN` in `apriltag_control/controller_v2/config.py`.
- [`tag_hover_controller.py`](drone_control/apriltag_control/tag_hover_controller.py) — older velocity-loop variant.
- `controller_v2/` — control logic + gains; `camera_setup/` — calibration; `setup/` — Ranger/stick test utilities.

---

## 7. Data path summary

| What | Where |
|---|---|
| TX12 flights (raw + merged) | `data_logging/recordings/<stamp>/` |
| Vicon-controller flights (raw + merged) | `drone_control/Vicon_control/flight_logs/<stamp>/` |
| TX12 calibration (shared) | `data_logging/tx12_joystick_cal.json` |
| Standalone recordings + synced | `pycode_ViCON/DataExchange/` |
| Blackbox decoder | `pycode_ViCON/tools/blackbox-tools/obj/blackbox_decode` |
| Camera calibration | `drone_control/apriltag_control/camera_setup/camera_calibration.npz` |

All recording/data folders are git-ignored (placeholder `.gitkeep` keeps the folders).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `VICON IS OFF` banner / `VICON:OFF` in status | No Vicon stream on :51001 at launch. Start Vicon Tracker (pointed at this laptop), then restart the script. |
| venv `exit 127` | Snap revision bumped; repoint `.venv/bin/python` + `pyvenv.cfg` at the new `~/snap/code/current/...` path. |
| "No Ranger serial port found" | Plug in the Ranger USB-C, or pass the port explicitly (`/dev/ttyUSB0 420000`). |
| Drone won't arm | Throttle must be at the bottom (< 1050 µs); arm is edge-gated (flip to DISARMED once first). |
| Mode switch not detected in calibration | The switch isn't exported as a joystick axis. Run `monitor_tx12.py`, flip it — if nothing moves, fix the EdgeTX *USB Joystick* config / move it to a channel ≤ a visible axis. |
| Video recording disabled | Run with `.venv/bin/python` (system python lacks `cv2`). |
| `combine` can't find a `.bbl` | Drop the flight's `.bbl` into `recordings/<stamp>/blackbox/`. |
| 3D trajectory blank | Hard-refresh the browser tab (and see the rendering note in §5). |

---

## 9. Command cheat-sheet

```bash
# --- collect (TX12) ---
.venv/bin/python data_logging/joystick_flight.py --calibrate        # one-time
.venv/bin/python data_logging/joystick_flight.py                    # fly + record
python3            data_logging/monitor_tx12.py                     # see every joystick input

# --- merge + view (TX12) ---
.venv/bin/python data_logging/combine_flight.py                     # newest session
.venv/bin/python data_logging/plot_flight.py                        # static PNG
.venv/bin/python data_logging/dashboard_flight.py                   # web dashboard

# --- standalone (no TX12) ---
.venv/bin/python pycode_ViCON/UdpReceiver_datacollection.py         # SPACE to record
.venv/bin/python pycode_ViCON/sync_log.py                           # merge w/ blackbox
.venv/bin/python pycode_ViCON/dashboard_synced.py                   # web dashboard

# --- autonomous Vicon hover (TX12 arms + records) ---
.venv/bin/python drone_control/Vicon_control/vicon_hover.py --dry-run  # verify control directions
.venv/bin/python drone_control/Vicon_control/vicon_hover.py            # fly (arm + flip record to launch)
.venv/bin/python drone_control/Vicon_control/combine.py               # merge newest flight
.venv/bin/python drone_control/Vicon_control/dashboard.py             # web dashboard (same UI)
```
