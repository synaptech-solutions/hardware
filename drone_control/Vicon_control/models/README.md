# Trained hover policies

Portable policy exports for `vicon_rl_hover.py` (numpy-only inference — see
`../rl_policy.py`). Each `.npz` is a betaflight-gym `GaussianPolicy` flattened by
`betaflight-gym/rl/export.py`: MLP weights + the learned obs-normalisation
mean/var + a metadata dict (obs layout, action mapping, target altitude).

## hover_acro.npz
Hover-in-place, **ACRO** (rate) mode. Pure obs→policy→action (no PID, no scripted
takeoff). The default model loaded by `vicon_rl_hover.py`.

- task: `hover`  |  drone_model: `air75_wisp`  |  obs_dim 22, act_dim 4 (AETR)
- net: tanh MLP [256, 256], target_alt 1.0 m
- obs: `pos_err(3) self_vel(3) rot_matrix(9) body_rates(3) last_action(4)` (body FRD)
- trained to global_step ~284.5M
- source: betaflight-gym `runs/hover/2026-06-16_15-13-59` @ git `26806fc53ad924b2a923c5fa0322e62b3880a6de`
