import torch
from torch import nn
from torch.nn import functional as F


class _MixlabLinear(nn.Module):
    """Linear projection using Mixlab's native [in, out] weight layout."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class _MixlabRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x):
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


def _rotate_pairs(values, phase):
    """Apply Mixlab's real 2x2 representation of the complex state rotation."""

    pairs = values.reshape(*values.shape[:-1], -1, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    cosine = torch.cos(phase)
    sine = torch.sin(phase)
    return torch.stack(
        (
            cosine * first + sine * second,
            -sine * first + cosine * second,
        ),
        dim=-1,
    ).flatten(-2)


class MixlabMamba3CanonicalBlock(nn.Module):
    """Inference reference for Mixlab's canonical Mamba-3 fused block.

    This deliberately mirrors the native scalar semantics instead of relying on
    a third-party Mamba kernel. The recurrent loop is sequential in time and
    vectorized across batch, channel, and state dimensions.
    """

    def __init__(self, config, block_config):
        super().__init__()
        model_dim = int(config.model_dim)
        inner_dim = int(block_config.get("inner_dim", 0) or model_dim)
        state_size = int(block_config.get("state_size", 0) or 16)
        n_groups = int(block_config.get("n_groups", 0) or 4)
        dt_rank = int(block_config.get("dt_rank", 0) or max(inner_dim // 16, 1))
        conv_kernel = int(block_config.get("conv_kernel", 0) or 4)
        use_conv = bool(block_config.get("use_conv", True))

        if inner_dim <= 0:
            raise ValueError("mamba3-canonical inner_dim must be positive")
        if state_size <= 0 or state_size % 2:
            raise ValueError("mamba3-canonical state_size must be positive and even")
        if n_groups <= 0 or inner_dim % n_groups:
            raise ValueError(
                "mamba3-canonical inner_dim must be divisible by n_groups"
            )
        if dt_rank <= 0:
            raise ValueError("mamba3-canonical dt_rank must be positive")
        if use_conv and conv_kernel <= 0:
            raise ValueError("mamba3-canonical conv_kernel must be positive")

        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.state_size = state_size
        self.n_groups = n_groups
        self.channels_per_group = inner_dim // n_groups
        self.dt_rank = dt_rank
        self.conv_kernel = conv_kernel
        self.use_conv = use_conv

        # The fused native block pins all three internal RMSNorm operations to
        # 1e-5 independently of the model-level final norm configuration.
        self.pre_norm = _MixlabRMSNorm(model_dim, eps=1e-5)
        self.w_x = _MixlabLinear(model_dim, inner_dim)
        if use_conv:
            self.conv_weight = nn.Parameter(torch.empty(inner_dim, conv_kernel))
            nn.init.xavier_uniform_(self.conv_weight)
        else:
            self.register_parameter("conv_weight", None)
        self.w_dt_low = _MixlabLinear(inner_dim, dt_rank)
        self.w_dt_high = _MixlabLinear(dt_rank, inner_dim)
        self.w_lambda_low = _MixlabLinear(inner_dim, dt_rank)
        self.w_lambda_high = _MixlabLinear(dt_rank, inner_dim)
        self.w_theta_low = _MixlabLinear(inner_dim, dt_rank)
        self.w_theta_high = _MixlabLinear(
            dt_rank, inner_dim * (state_size // 2)
        )
        self.w_B = _MixlabLinear(inner_dim, n_groups * state_size)
        self.w_C = _MixlabLinear(inner_dim, n_groups * state_size)
        self.B_norm = _MixlabRMSNorm(state_size, eps=1e-5)
        self.C_norm = _MixlabRMSNorm(state_size, eps=1e-5)
        self.B_bias = nn.Parameter(torch.ones(n_groups * state_size))
        self.C_bias = nn.Parameter(torch.ones(n_groups * state_size))
        self.A_log = nn.Parameter(torch.empty(inner_dim, state_size))
        self.dt_bias = nn.Parameter(torch.zeros(inner_dim))
        self.post_norm = _MixlabRMSNorm(inner_dim, eps=1e-5)
        self.w_gate = _MixlabLinear(model_dim, inner_dim)
        self.w_out = _MixlabLinear(inner_dim, model_dim)

    def _causal_depthwise_conv(self, x):
        # torch.conv1d is cross-correlation. Flip K so its output at t is
        # sum_k x[t-k] * weight[k], matching Mixlab's native causal primitive.
        channels_first = x.transpose(1, 2)
        padded = F.pad(channels_first, (self.conv_kernel - 1, 0))
        kernel = self.conv_weight.flip(-1).unsqueeze(1)
        return F.conv1d(
            padded,
            kernel,
            groups=self.inner_dim,
        ).transpose(1, 2)

    def _selective_scan(
        self,
        x,
        dt_raw,
        lambda_raw,
        theta,
        b_projected,
        c_projected,
        validity_mask,
    ):
        batch, steps, _ = x.shape
        pairs = self.state_size // 2
        delta = F.softplus(dt_raw)
        interpolation = torch.sigmoid(lambda_raw)
        theta = theta.reshape(batch, steps, self.inner_dim, pairs)
        b_projected = b_projected.reshape(
            batch, steps, self.n_groups, self.state_size
        )
        c_projected = c_projected.reshape(
            batch, steps, self.n_groups, self.state_size
        )

        decay = -torch.exp(self.A_log).unsqueeze(0)
        state = x.new_zeros((batch, self.inner_dim, self.state_size))
        phase = x.new_zeros((batch, self.inner_dim, pairs))
        previous_b = x.new_zeros((batch, self.inner_dim, self.state_size))
        previous_x = x.new_zeros((batch, self.inner_dim, 1))
        outputs = []

        for step in range(steps):
            valid = validity_mask[:, step].reshape(batch, 1, 1)
            delta_step = delta[:, step]
            lambda_step = interpolation[:, step]
            candidate_phase = phase + delta_step.unsqueeze(-1) * theta[:, step]
            phase = torch.where(valid, candidate_phase, phase)

            b_step = b_projected[:, step].repeat_interleave(
                self.channels_per_group, dim=1
            )
            c_step = c_projected[:, step].repeat_interleave(
                self.channels_per_group, dim=1
            )
            b_rotated = _rotate_pairs(b_step, phase)
            c_rotated = _rotate_pairs(c_step, phase)

            alpha = torch.exp(delta_step.unsqueeze(-1) * decay)
            beta = (
                (1.0 - lambda_step).unsqueeze(-1)
                * delta_step.unsqueeze(-1)
                * alpha
            )
            gamma = lambda_step.unsqueeze(-1) * delta_step.unsqueeze(-1)
            current = gamma * b_rotated * x[:, step].unsqueeze(-1)
            previous = beta * previous_b * previous_x
            candidate_state = alpha * state + previous + current
            state = torch.where(valid, candidate_state, state)

            output = torch.sum(state * c_rotated, dim=-1)
            outputs.append(torch.where(valid.squeeze(-1), output, 0.0))
            previous_b = torch.where(valid, b_rotated, previous_b)
            previous_x = torch.where(valid, x[:, step].unsqueeze(-1), previous_x)

        return torch.stack(outputs, dim=1)

    def forward(self, x, attention_mask=None, dwa=None):
        if x.ndim != 3:
            raise ValueError("mamba3-canonical expects [batch, sequence, hidden]")
        batch, steps, _ = x.shape
        if steps <= 0:
            raise ValueError("mamba3-canonical requires at least one token")
        if attention_mask is None:
            validity_mask = torch.ones(
                (batch, steps), dtype=torch.bool, device=x.device
            )
        else:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
                batch,
                steps,
            ):
                raise ValueError(
                    "mamba3-canonical attention_mask must have shape "
                    f"({batch}, {steps})"
                )
            validity_mask = attention_mask.to(device=x.device).ne(0)

        masked_x = x * validity_mask.to(dtype=x.dtype).unsqueeze(-1)
        x_norm = self.pre_norm(masked_x)
        x_branch = self.w_x(x_norm)
        if self.use_conv:
            x_branch = self._causal_depthwise_conv(x_branch)

        dt = self.w_dt_high(self.w_dt_low(x_branch)) + self.dt_bias
        interpolation = self.w_lambda_high(self.w_lambda_low(x_branch))
        theta = self.w_theta_high(self.w_theta_low(x_branch))
        b_projected = self.w_B(x_branch)
        c_projected = self.w_C(x_branch)
        b_projected = self.B_norm(
            b_projected.reshape(-1, self.state_size)
        ).reshape(batch, steps, -1) + self.B_bias
        c_projected = self.C_norm(
            c_projected.reshape(-1, self.state_size)
        ).reshape(batch, steps, -1) + self.C_bias

        scanned = self._selective_scan(
            x_branch,
            dt,
            interpolation,
            theta,
            b_projected,
            c_projected,
            validity_mask,
        )
        gated = self.post_norm(scanned) * F.silu(self.w_gate(x_norm))
        output = masked_x + self.w_out(gated)
        output = output * validity_mask.to(dtype=output.dtype).unsqueeze(-1)
        if dwa is not None:
            output = dwa.apply(output)
        return output
