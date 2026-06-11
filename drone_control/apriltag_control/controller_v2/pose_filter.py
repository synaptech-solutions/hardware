"""Per-axis (fwd, lat, up) pose filter — LPF on tag position with hold-on-loss,
velocity from filtered-position delta. Lifted from tag_hover_controller.py
(flight-proven). See that file's PoseFilter docstring for why we don't
dead-reckon position across vision dropouts (was the snap-back failure mode).
"""
import math

from . import config


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PoseFilter:
    def __init__(self):
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.last_t = None
        self.last_fresh_t = None
        self.inited = False

    def reset(self, p0, t_now):
        self.p = list(p0)
        self.v = [0.0, 0.0, 0.0]
        self.last_t = t_now
        self.last_fresh_t = t_now
        self.inited = True

    def update(self, vis_pos, vis_fresh, t_now, ground_pinned=False):
        if not self.inited:
            if vis_fresh:
                self.reset(vis_pos, t_now)
            return
        dt = _clamp(t_now - (self.last_t or t_now), 1e-3,
                    config.ESTIMATOR_MAX_DT_S)
        if vis_fresh:
            blind = (self.last_fresh_t is None
                     or (t_now - self.last_fresh_t) > config.RESEED_AFTER_LOSS_S)
            if blind:
                self.p = list(vis_pos)
                self.v = [0.0, 0.0, 0.0]
            else:
                for i in range(3):
                    p_new = ((1.0 - config.ALPHA_POS) * self.p[i]
                             + config.ALPHA_POS * vis_pos[i])
                    v_inst = (p_new - self.p[i]) / dt
                    self.v[i] = ((1.0 - config.ALPHA_VEL) * self.v[i]
                                 + config.ALPHA_VEL * v_inst)
                    self.p[i] = p_new
            self.last_fresh_t = t_now
        else:
            decay = math.exp(-dt / config.VEL_DECAY_TAU_S)
            for i in range(3):
                self.v[i] *= decay
        if ground_pinned:
            self.v = [0.0, 0.0, 0.0]
        self.last_t = t_now

    def age(self, now):
        if self.last_fresh_t is None:
            return 1e9
        return now - self.last_fresh_t
