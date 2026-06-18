# Trained hover policies

Portable policy exports for `vicon_rl_hover.py` (numpy-only inference — see
`../rl_policy.py`). Each `.npz` is a betaflight-gym `GaussianPolicy` flattened by
`betaflight-gym/rl/export.py`: MLP weights + the learned obs-normalisation
mean/var + a metadata dict (obs layout, action mapping, target altitude).

## hover_acro.npz
Hover-in-place, **ACRO** (rate) mode. Pure obs→policy→action (no PID, no scripted
takeoff). The default model loaded by `vicon_rl_hover.py`.

- task: `hover`  |  drone_model: `air75_wisp`  |  obs_dim 23, act_dim 4 (AETR)
- net: tanh MLP [256, 256], target_alt 1.0 m
- obs: `pos_err(3) self_vel(3) rot_matrix(9) body_rates(3) last_action(4) mass(1)` (body FRD)
- **mass-conditioned**: the obs ends with the measured all-up mass (kg). Exported
  `mass_kg = 0.0344` (the domain-randomisation centre, range 0.0275–0.0413 kg);
  `HoverPolicyController` feeds it automatically. Pass the real weighed mass via
  `mass_kg=` if a given airframe differs.
- **control rate is read from the model**: `vicon_rl_hover.py` drives its loop
  period from the policy's `control_dt` meta (this model: `0.02` = 50 Hz), so each
  model deploys at its own trained rate — the policy is the sole, non-dt-aware
  stabiliser and must run at the rate it learned. (NOT `channels.TX_HZ` = 100 Hz,
  which the PID flight scripts use.)
- trained to ~2.13B env-steps (the npz `global_step` field is an int32 overflow at
  this count — ignore it)
- source: betaflight-gym `runs/hover/2026-06-18_03-03-54` @ git `dd36ce94b622b2e1b336c9cc8668712eb2948702`
