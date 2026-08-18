import math

import torch
from torch import nn
from torch.nn import functional as F


class MixlabS4DBlock(nn.Module):
    """Fixed-length PyTorch inference reference for Mixlab's S4D block."""

    def __init__(self, config, block_config, norm_factory):
        super().__init__()
        dim = int(config.model_dim)
        state_size = int(block_config.get("state_size", 0) or 64)
        n_ssm = int(block_config.get("n_ssm", 0) or dim)
        if state_size <= 0 or state_size % 2:
            raise ValueError("s4d state_size must be positive and even")
        if n_ssm <= 0 or dim % n_ssm:
            raise ValueError("s4d n_ssm must divide model_dim")

        self.dim = dim
        self.state_pairs = state_size // 2
        self.n_ssm = n_ssm
        self.channels_per_ssm = dim // n_ssm
        self.bidirectional = bool(block_config.get("bidirectional", False))
        self.trainable_b = bool(block_config.get("trainable_b", False))
        self.discretization = str(
            block_config.get("discretization", "zoh") or "zoh"
        ).lower()
        if self.discretization not in ("zoh", "bilinear"):
            raise ValueError(f"unsupported s4d discretization {self.discretization!r}")
        self.output_transform = str(
            block_config.get("output_transform", "none") or "none"
        ).lower()
        if self.output_transform not in ("none", "glu"):
            raise ValueError(f"unsupported s4d output_transform {self.output_transform!r}")
        self.norm_placement = str(
            getattr(config, "norm_placement", "pre") or "pre"
        ).lower()
        if self.norm_placement not in ("pre", "post", "post_residual", "sandwich"):
            raise ValueError(f"unsupported s4d norm_placement {self.norm_placement!r}")

        self.norm = (
            norm_factory(config, dim)
            if self.norm_placement in ("pre", "sandwich")
            else None
        )
        self.log_dt = nn.Parameter(torch.empty(dim))
        self.log_A_real = nn.Parameter(torch.empty(n_ssm, self.state_pairs))
        self.A_imag = nn.Parameter(torch.empty(n_ssm, self.state_pairs))
        if self.trainable_b:
            self.B_real = nn.Parameter(torch.empty(n_ssm, self.state_pairs))
            self.B_imag = nn.Parameter(torch.empty(n_ssm, self.state_pairs))
        else:
            self.register_parameter("B_real", None)
            self.register_parameter("B_imag", None)
        self.C_real = nn.Parameter(torch.empty(dim, self.state_pairs))
        self.C_imag = nn.Parameter(torch.empty(dim, self.state_pairs))
        if self.bidirectional:
            self.C_backward_real = nn.Parameter(torch.empty(dim, self.state_pairs))
            self.C_backward_imag = nn.Parameter(torch.empty(dim, self.state_pairs))
        else:
            self.register_parameter("C_backward_real", None)
            self.register_parameter("C_backward_imag", None)
        self.D = nn.Parameter(torch.empty(dim))
        sobolev_config = block_config.get("sobolev_filter")
        if sobolev_config:
            if isinstance(sobolev_config, dict):
                beta_init = float(sobolev_config.get("beta_init", 0.0) or 0.0)
                granularity = str(
                    sobolev_config.get("granularity", "channel") or "channel"
                )
                bounds = sobolev_config.get("bounds")
                trainable = bool(sobolev_config.get("trainable", True))
            else:
                beta_init = 0.0
                granularity = "channel"
                bounds = None
                trainable = True
            beta_size = 1 if granularity == "layer" else dim
            if bounds is not None:
                lower, upper = float(bounds[0]), float(bounds[1])
                ratio = (beta_init - (lower + upper) / 2.0) / (
                    (upper - lower) / 2.0
                )
                beta_init = math.atanh(ratio)
            self.sobolev_beta = nn.Parameter(
                torch.full((beta_size,), beta_init), requires_grad=trainable
            )
            self.sobolev_bounds = bounds
        else:
            self.register_parameter("sobolev_beta", None)
            self.sobolev_bounds = None
        self.out_proj = (
            _MixlabS4DLinear(dim, 2 * dim, bias=True)
            if self.output_transform == "glu"
            else None
        )
        self.post_norm = (
            norm_factory(config, dim)
            if self.norm_placement in ("post", "sandwich")
            else None
        )
        self.post_residual_norm = (
            norm_factory(config, dim)
            if self.norm_placement == "post_residual"
            else None
        )
        self.dropout_probability = float(
            getattr(config, "hidden_dropout", 0.0) or 0.0
        )
        self.tie_dropout = bool(block_config.get("tie_dropout", False))

    def _broadcast_groups(self, value):
        # Match state-spaces/s4's einops repeat("t n -> (v t) n"): group indices
        # interleave over model channels, so channel c uses group c % n_ssm.
        # repeat_interleave would assign contiguous channel blocks instead.
        return value.repeat(self.channels_per_ssm, 1)

    def _discretize(self):
        dt = torch.exp(self.log_dt.float()).reshape(self.dim, 1)
        a_real = -torch.exp(self._broadcast_groups(self.log_A_real.float()))
        a_imag = self._broadcast_groups(self.A_imag.float())
        if self.trainable_b:
            b_real = self._broadcast_groups(self.B_real.float())
            b_imag = self._broadcast_groups(self.B_imag.float())
        else:
            b_real = torch.ones_like(a_real)
            b_imag = torch.zeros_like(a_imag)

        dt_a_real = dt * a_real
        dt_a_imag = dt * a_imag
        if self.discretization == "zoh":
            magnitude = torch.exp(dt_a_real)
            abar_real = magnitude * torch.cos(dt_a_imag)
            abar_imag = magnitude * torch.sin(dt_a_imag)
            numerator_real = abar_real - 1.0
            denominator = a_real.square() + a_imag.square()
            base_real = (
                numerator_real * a_real + abar_imag * a_imag
            ) / denominator
            base_imag = (
                abar_imag * a_real - numerator_real * a_imag
            ) / denominator
            bbar_real = base_real * b_real - base_imag * b_imag
            bbar_imag = base_real * b_imag + base_imag * b_real
        else:
            half_real = 0.5 * dt_a_real
            half_imag = 0.5 * dt_a_imag
            numerator_real = 1.0 + half_real
            numerator_imag = half_imag
            denominator_real = 1.0 - half_real
            denominator_imag = -half_imag
            denominator = denominator_real.square() + denominator_imag.square()
            abar_real = (
                numerator_real * denominator_real
                + numerator_imag * denominator_imag
            ) / denominator
            abar_imag = (
                numerator_imag * denominator_real
                - numerator_real * denominator_imag
            ) / denominator
            dt_b_real = dt * b_real
            dt_b_imag = dt * b_imag
            bbar_real = (
                dt_b_real * denominator_real + dt_b_imag * denominator_imag
            ) / denominator
            bbar_imag = (
                dt_b_imag * denominator_real - dt_b_real * denominator_imag
            ) / denominator

        magnitude_sq = torch.clamp_min(abar_real.square() + abar_imag.square(), 1e-30)
        log_magnitude = 0.5 * torch.log(magnitude_sq)
        phase = torch.atan2(abar_imag, abar_real)
        return bbar_real, bbar_imag, log_magnitude, phase

    def _kernel(self, c_real, c_imag, length):
        bbar_real, bbar_imag, log_magnitude, phase = self._discretize()
        gamma_real = c_real.float() * bbar_real - c_imag.float() * bbar_imag
        gamma_imag = c_real.float() * bbar_imag + c_imag.float() * bbar_real
        positions = torch.arange(
            length, device=gamma_real.device, dtype=torch.float32
        ).reshape(1, 1, length)
        magnitude_powers = torch.exp(log_magnitude.unsqueeze(2) * positions)
        phases = phase.unsqueeze(2) * positions
        power_real = magnitude_powers * torch.cos(phases)
        power_imag = magnitude_powers * torch.sin(phases)
        terms = (
            gamma_real.unsqueeze(2) * power_real
            - gamma_imag.unsqueeze(2) * power_imag
        )
        return 2.0 * terms.sum(dim=1)

    def _convolve(self, x):
        steps = x.shape[1]
        forward_kernel = self._kernel(self.C_real, self.C_imag, steps)
        if self.bidirectional:
            backward_kernel = self._kernel(
                self.C_backward_real, self.C_backward_imag, steps
            )
            zeros = torch.zeros_like(forward_kernel)
            kernel = torch.cat((forward_kernel, zeros), dim=1) + torch.cat(
                (zeros, backward_kernel.flip(1)), dim=1
            )
            fft_len = 2 * steps
        else:
            kernel = forward_kernel
            fft_len = 1 << max(1, 2 * steps - 1).bit_length()
        x_frequency = torch.fft.rfft(x.float(), n=fft_len, dim=1)
        kernel_frequency = torch.fft.rfft(kernel, n=fft_len, dim=1)
        frequency_product = x_frequency * kernel_frequency.transpose(0, 1).unsqueeze(0)
        if self.sobolev_beta is not None:
            frequency_bins = frequency_product.shape[1]
            normalized_frequency = (
                torch.arange(
                    frequency_bins,
                    device=frequency_product.device,
                    dtype=torch.float32,
                )
                / float(fft_len)
            ).reshape(1, frequency_bins, 1)
            effective_beta = self.sobolev_beta.float()
            if self.sobolev_bounds is not None:
                lower, upper = self.sobolev_bounds
                effective_beta = (
                    (float(lower) + float(upper)) / 2.0
                    + (float(upper) - float(lower))
                    / 2.0
                    * torch.tanh(effective_beta)
                )
            if effective_beta.numel() == 1:
                effective_beta = effective_beta.expand(self.dim)
            frequency_filter = torch.pow(
                1.0 + normalized_frequency,
                effective_beta.reshape(1, 1, self.dim),
            )
            frequency_product = frequency_product * frequency_filter
        convolved = torch.fft.irfft(
            frequency_product,
            n=fft_len,
            dim=1,
        )[:, :steps, :]
        return convolved + x.float() * self.D.float().reshape(1, 1, self.dim)

    def _dropout(self, x):
        if not self.training or self.dropout_probability <= 0.0:
            return x
        if not self.tie_dropout:
            return F.dropout(x, p=self.dropout_probability, training=True)
        keep = 1.0 - self.dropout_probability
        mask = torch.empty(
            (x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype
        ).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x, attention_mask=None, dwa=None):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("s4d expects [batch, sequence, model_dim]")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(x.shape[:2]):
                raise ValueError("s4d attention_mask must match [batch, sequence]")
            if not bool(torch.all(attention_mask.ne(0))):
                raise ValueError(
                    "continuous S4D HF export requires fixed unpadded records"
                )

        branch_input = self.norm(x) if self.norm is not None else x
        delta = self._convolve(branch_input)
        if self.output_transform == "glu":
            delta = F.gelu(delta, approximate="none")
            delta = self._dropout(delta)
            projected = self.out_proj(delta)
            value, gate = projected.split(self.dim, dim=-1)
            delta = value * torch.sigmoid(gate)
        else:
            delta = F.gelu(delta, approximate="tanh")
        if self.post_norm is not None:
            delta = self.post_norm(delta)
        delta = self._dropout(delta)
        output = x + delta
        if self.post_residual_norm is not None:
            output = self.post_residual_norm(output)
        if dwa is not None:
            output = dwa.apply(output)
        return output


class _MixlabS4DLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x):
        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output
