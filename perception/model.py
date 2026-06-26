"""GateNet segmentation U-Net (TU Delft MonoRace, arXiv:2601.15222).

Reconstructed from the architecture table in the paper. A U-Net encoder-decoder
with channel counts scaled by 1/f, **additive** skip connections, and five
multi-scale single-channel outputs for deep supervision:

    Encoder            Decoder        Outputs
    inc-64/f      ->  up4-64/f   -> outc4-1   (highest resolution)
    down1-128/f   ->  up3-64/f   -> outc3-1
    down2-256/f   ->  up2-128/f  -> outc2-1
    down3-512/f   ->  up1-256/f  -> outc1-1
    down4-512/f                  -> outc0-1   (bottleneck, lowest resolution)

The paper specifies additive skips but not how channel mismatches are bridged; we
make each up-block's transposed conv emit the skip's channel count, add, then apply
a double-conv to the labelled output width. Outputs are returned as logits ordered
[y0 (coarsest) .. y4 (finest)]; deployment uses y4.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

# Nominal channel widths (before the 1/f scaling) per the paper's table.
_ENC = (64, 128, 256, 512, 512)  # inc, down1..down4
_DEC = (256, 128, 64, 64)  # up1..up4 output widths

_INIT = nn.initializers.xavier_uniform()  # paper uses Xavier uniform


class DoubleConv(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        for _ in range(2):
            x = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False, kernel_init=_INIT)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
        return x


class GateNetUNet(nn.Module):
    """Multi-scale gate-segmentation U-Net. ``f`` divides all channel widths."""

    f: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> list[jnp.ndarray]:
        enc = [c // self.f for c in _ENC]
        dec = [c // self.f for c in _DEC]

        # Encoder (store skips s0..s3; s4 is the bottleneck).
        s0 = DoubleConv(enc[0], name="inc")(x, train)
        skips = [s0]
        h = s0
        for i, ch in enumerate(enc[1:], start=1):
            h = nn.max_pool(h, (2, 2), strides=(2, 2))
            h = DoubleConv(ch, name=f"down{i}")(h, train)
            skips.append(h)
        bottleneck = skips.pop()  # s4

        outputs = [nn.Conv(1, (1, 1), kernel_init=_INIT, name="outc0")(bottleneck)]  # coarsest

        # Decoder: up1..up4. Transposed conv emits skip channels, add skip, double-conv.
        h = bottleneck
        for j, out_ch in enumerate(dec, start=1):
            skip = skips[-j]  # s3, s2, s1, s0
            h = nn.ConvTranspose(
                skip.shape[-1], (2, 2), strides=(2, 2), use_bias=False,
                kernel_init=_INIT, name=f"up{j}_tconv",
            )(h)
            h = nn.BatchNorm(use_running_average=not train, name=f"up{j}_bn")(h)
            h = h + skip
            h = DoubleConv(out_ch, name=f"up{j}")(h, train)
            outputs.append(nn.Conv(1, (1, 1), kernel_init=_INIT, name=f"outc{j}")(h))

        return outputs  # [y0 coarsest .. y4 finest]
