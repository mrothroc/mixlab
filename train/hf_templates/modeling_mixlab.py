import math
import os
import struct

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_mixlab import MixlabConfig


class MixlabRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class MixlabLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, self.weight)


def rotate_adjacent_rope(x, rope_dims, cos_t=None, sin_t=None, base=10000.0):
    head_dim = x.shape[-1]
    if rope_dims is None or rope_dims <= 0 or rope_dims >= head_dim:
        rope_dims = head_dim
    if rope_dims % 2 != 0:
        raise ValueError(f"rope_dims must be even, got {rope_dims}")

    seq_len = x.shape[-2]
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    pair_count = rope_dims // 2
    if cos_t is None or sin_t is None:
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        dim_idx = torch.arange(pair_count, device=x.device, dtype=x.dtype)
        freqs = torch.exp(dim_idx * (-math.log(base) * 2.0 / float(rope_dims)))
        angles = positions[:, None] * freqs[None, :]
        cos_t = torch.cos(angles).view(1, 1, seq_len, pair_count)
        sin_t = torch.sin(angles).view(1, 1, seq_len, pair_count)
    else:
        cos_t = cos_t[:, :, :seq_len, :]
        sin_t = sin_t[:, :, :seq_len, :]

    pairs = x_rot.view(*x_rot.shape[:-1], pair_count, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = torch.stack((even * cos_t - odd * sin_t, even * sin_t + odd * cos_t), dim=-1)
    rotated = rotated.reshape(*x_rot.shape)
    if x_pass.shape[-1] == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


class MixlabPlainBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        heads = int(block_config["heads"])
        self.heads = heads
        self.kv_heads = int(block_config.get("kv_heads", 0) or heads)
        self.head_dim = dim // heads
        if heads % self.kv_heads != 0:
            raise ValueError(f"heads must be divisible by kv_heads, got heads={heads} kv_heads={self.kv_heads}")
        if int(block_config.get("kv_source", 0) or 0) > 0:
            raise ValueError("kv_source is not supported by Mixlab HF export")
        self.kv_group_size = heads // self.kv_heads
        self.rope_dims = int(block_config.get("rope_dims", 0) or 0)
        self.effective_rope_dims = self.rope_dims
        if self.effective_rope_dims <= 0 or self.effective_rope_dims >= self.head_dim:
            self.effective_rope_dims = self.head_dim
        self.attention_mask = str(block_config.get("attention_mask", "") or "causal").lower()
        self.window_size = int(block_config.get("window_size", 0) or 0)
        self.relative_attention = str(block_config.get("relative_attention", "") or "none").lower()
        self.relative_attention_window = int(block_config.get("relative_attention_window", 0) or 128)
        self.norm = MixlabRMSNorm(dim)
        self.wq = MixlabLinear(dim, dim)
        self.wk = MixlabLinear(dim, self.kv_heads * self.head_dim)
        self.wv = MixlabLinear(dim, self.kv_heads * self.head_dim)
        self.q_norm = None
        self.k_norm = None
        if bool(block_config.get("qk_norm", False)):
            self.q_norm = MixlabRMSNorm(self.head_dim)
            self.k_norm = MixlabRMSNorm(self.head_dim)
        self.relative_embeddings = None
        self.w_pos_key = None
        self.w_pos_query = None
        if self.relative_attention == "deberta_p2c_c2p":
            rel_rows = 2 * self.relative_attention_window
            self.relative_embeddings = nn.Parameter(torch.empty(rel_rows, dim))
            self.w_pos_key = MixlabLinear(dim, dim)
            self.w_pos_query = MixlabLinear(dim, dim)
            nn.init.xavier_uniform_(self.relative_embeddings)
        elif self.relative_attention not in ("", "none"):
            raise ValueError(f"unsupported exported relative_attention {self.relative_attention!r}")
        self.qk_gain = None
        if float(block_config.get("qk_gain", 0.0) or 0.0) > 0.0:
            self.qk_gain = nn.Parameter(torch.empty(heads))
            nn.init.constant_(self.qk_gain, float(block_config["qk_gain"]))
        self.wo = MixlabLinear(dim, dim)
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.ff1 = MixlabLinear(dim, ffn_dim)
        self.ff2 = MixlabLinear(ffn_dim, dim)
        mask = self._build_attention_mask(config.seq_len, self.window_size)
        self.register_buffer("causal_mask", mask.view(1, 1, config.seq_len, config.seq_len), persistent=False)
        pair_count = self.effective_rope_dims // 2
        positions = torch.arange(config.seq_len, dtype=torch.float32)
        dim_idx = torch.arange(pair_count, dtype=torch.float32)
        freqs = torch.exp(dim_idx * (-math.log(10000.0) * 2.0 / float(self.effective_rope_dims)))
        angles = positions[:, None] * freqs[None, :]
        self.register_buffer("rope_cos", torch.cos(angles).view(1, 1, config.seq_len, pair_count), persistent=False)
        self.register_buffer("rope_sin", torch.sin(angles).view(1, 1, config.seq_len, pair_count), persistent=False)

    @staticmethod
    def _build_attention_mask(seq_len, window_size):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        if window_size > 0 and window_size < seq_len:
            pos = torch.arange(seq_len)
            query_pos = pos.view(seq_len, 1)
            key_pos = pos.view(1, seq_len)
            too_old = key_pos < (query_pos - (window_size - 1))
            mask = mask | too_old
        return mask

    def _repeat_kv(self, x):
        if self.kv_group_size == 1:
            return x
        bsz, kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, kv_heads, self.kv_group_size, seq_len, head_dim)
        return x.reshape(bsz, self.heads, seq_len, head_dim)

    def _deberta_relative_bias(self, q, k, seq_len):
        rel_key = self.w_pos_key(self.relative_embeddings)
        rel_query = self.w_pos_query(self.relative_embeddings)
        rel_key = rel_key.view(2 * self.relative_attention_window, self.heads, self.head_dim).permute(1, 0, 2)
        rel_query = rel_query.view(2 * self.relative_attention_window, self.heads, self.head_dim).permute(1, 0, 2)
        pos = torch.arange(seq_len, device=q.device)
        c2p = torch.clamp(
            pos.view(seq_len, 1) - pos.view(1, seq_len) + self.relative_attention_window,
            0,
            2 * self.relative_attention_window - 1,
        )
        p2c = torch.clamp(
            pos.view(1, seq_len) - pos.view(seq_len, 1) + self.relative_attention_window,
            0,
            2 * self.relative_attention_window - 1,
        )
        c2p_key = rel_key[:, c2p, :]
        p2c_query = rel_query[:, p2c, :]
        c2p_bias = torch.einsum("bhid,hijd->bhij", q, c2p_key)
        p2c_bias = torch.einsum("bhjd,hijd->bhij", k, p2c_query)
        return c2p_bias + p2c_bias

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        bsz, seq_len, dim = x_norm.shape

        q = self.wq(x_norm).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_norm).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_norm).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.relative_attention == "deberta_p2c_c2p":
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + self._deberta_relative_bias(q, k, seq_len)
            scores = scores / math.sqrt(float(self.head_dim * 3))
        else:
            q = rotate_adjacent_rope(q, self.effective_rope_dims, self.rope_cos, self.rope_sin)
            k = rotate_adjacent_rope(k, self.effective_rope_dims, self.rope_cos, self.rope_sin)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        if self.qk_gain is not None:
            scores = scores * self.qk_gain.view(1, self.heads, 1, 1)
        if self.attention_mask == "causal":
            causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(causal_mask, -1e9)
        elif self.attention_mask in ("bidirectional", "none"):
            pass
        else:
            raise ValueError(f"unsupported exported attention_mask {self.attention_mask!r}")
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = residual + self.wo(ctx)

        ff = self.ff2(torch.nn.functional.silu(self.ff1(x)))
        return x + ff


class MixlabMoEExpert(nn.Module):
    def __init__(self, config, expert_config):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.expert_type = str(expert_config.get("type", "") or "swiglu").lower()
        self.activation = str(expert_config.get("activation", "") or "silu").lower()
        self.leaky_slope = float(expert_config.get("leaky_slope", 0.0) or 0.5)
        if self.expert_type in ("swiglu", "geglu"):
            self.w_gate = MixlabLinear(dim, ffn_dim)
            self.w_up = MixlabLinear(dim, ffn_dim)
            self.w_down = MixlabLinear(ffn_dim, dim)
        elif self.expert_type == "mlp":
            self.w_up = MixlabLinear(dim, ffn_dim)
            self.w_down = MixlabLinear(ffn_dim, dim)
        else:
            raise ValueError(f"unsupported exported MoE expert type {self.expert_type!r}")

    def forward(self, x):
        if self.expert_type in ("swiglu", "geglu"):
            gate = self.w_gate(x)
            if self.expert_type == "swiglu":
                gate = torch.sigmoid(gate)
            else:
                gate = torch.nn.functional.gelu(gate)
            return self.w_down(gate * self.w_up(x))

        up = self.w_up(x)
        if self.activation == "silu":
            act = torch.nn.functional.silu(up)
        elif self.activation == "gelu":
            act = torch.nn.functional.gelu(up)
        elif self.activation == "relu":
            act = torch.relu(up)
        elif self.activation == "leaky_relu_sq":
            act = torch.nn.functional.leaky_relu(up, negative_slope=self.leaky_slope)
            act = act * act
        else:
            raise ValueError(f"unsupported exported MoE MLP activation {self.activation!r}")
        return self.w_down(act)


class MixlabMoEBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        self.num_experts = int(block_config["num_experts"])
        self.top_k = int(block_config.get("top_k", 0) or min(2, self.num_experts))
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1,num_experts], got {self.top_k}")
        router = str(block_config.get("router", "") or "linear").lower()
        if router != "linear":
            raise ValueError(f"unsupported exported MoE router {router!r}")
        self.norm = MixlabRMSNorm(dim)
        self.router_w = nn.Parameter(torch.empty(dim, self.num_experts))
        nn.init.xavier_uniform_(self.router_w)
        expert_config = block_config.get("expert_block") or {"type": "swiglu"}
        self.experts = nn.ModuleList([MixlabMoEExpert(config, expert_config) for _ in range(self.num_experts)])

    def forward(self, x):
        x_norm = self.norm(x)
        router_logits = torch.matmul(x_norm, self.router_w)
        probs = torch.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_probs = top_probs / torch.clamp(top_probs.sum(dim=-1, keepdim=True), min=1e-12)
        dispatch = torch.zeros_like(probs).scatter(-1, top_idx, top_probs)
        flat_x = x_norm.reshape(-1, x_norm.shape[-1])
        flat_dispatch = dispatch.reshape(-1, self.num_experts)
        flat_delta = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            selected = torch.nonzero(flat_dispatch[:, expert_id] > 0, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            expert_out = expert(flat_x.index_select(0, selected))
            weight = flat_dispatch.index_select(0, selected)[:, expert_id].unsqueeze(-1)
            flat_delta.index_add_(0, selected, expert_out * weight)
        return x + flat_delta.view_as(x)


class MixlabSwiGLUBlock(nn.Module):
    def __init__(self, config, gate_activation="sigmoid"):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.gate_activation = gate_activation
        self.norm = MixlabRMSNorm(dim)
        self.w_gate = MixlabLinear(dim, ffn_dim)
        self.w_up = MixlabLinear(dim, ffn_dim)
        self.w_down = MixlabLinear(ffn_dim, dim)

    def forward(self, x):
        x_norm = self.norm(x)
        gate = self.w_gate(x_norm)
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(gate)
        elif self.gate_activation == "gelu":
            gate = torch.nn.functional.gelu(gate)
        else:
            raise ValueError(f"unsupported exported gate activation {self.gate_activation!r}")
        gated = gate * self.w_up(x_norm)
        return x + self.w_down(gated)


class MixlabMLPBlock(nn.Module):
    def __init__(self, config, block_config):
        super().__init__()
        dim = config.model_dim
        ffn_dim = max(dim, int(round(dim * float(config.mlp_mult))))
        self.activation = str(block_config.get("activation", "") or "silu").lower()
        self.leaky_slope = float(block_config.get("leaky_slope", 0.0) or 0.5)
        self.norm = MixlabRMSNorm(dim)
        self.w_up = MixlabLinear(dim, ffn_dim)
        self.w_down = MixlabLinear(ffn_dim, dim)

    def forward(self, x):
        x_norm = self.norm(x)
        up = self.w_up(x_norm)
        if self.activation == "silu":
            act = torch.nn.functional.silu(up)
        elif self.activation == "gelu":
            act = torch.nn.functional.gelu(up)
        elif self.activation == "relu":
            act = torch.relu(up)
        elif self.activation == "leaky_relu_sq":
            act = torch.nn.functional.leaky_relu(up, negative_slope=self.leaky_slope)
            act = act * act
        else:
            raise ValueError(f"unsupported exported MLP activation {self.activation!r}")
        return x + self.w_down(act)


def load_char_lookup(config):
    if int(getattr(config, "char_vocab_size", 0) or 0) <= 0:
        return None
    filename = getattr(config, "char_features_file", "") or "char_features.bin"
    base = getattr(config, "_name_or_path", "") or ""
    path = os.path.join(base, filename) if base else filename
    if not os.path.exists(path):
        raise ValueError(f"char features enabled but {path!r} is missing")
    vocab = int(config.vocab_size)
    char_vocab = int(config.char_vocab_size)
    slots = int(config.char_max_per_token)
    header_ints = 256
    header_bytes = header_ints * 4
    want = header_bytes + vocab * slots * 2
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) != want:
        raise ValueError(f"char feature file size={len(blob)} want={want}")
    header = struct.unpack("<" + "i" * header_ints, blob[:header_bytes])
    if header[0] != 20260526 or header[1] != 1:
        raise ValueError("invalid char feature magic/version")
    if header[2] != vocab or header[3] != char_vocab or header[4] != slots:
        raise ValueError("char feature metadata does not match config")
    payload = torch.tensor(
        struct.unpack("<" + "H" * (vocab * slots), blob[header_bytes:]),
        dtype=torch.long,
    )
    return payload.view(vocab, slots)


class MixlabForCausalLM(PreTrainedModel):
    config_class = MixlabConfig
    base_model_prefix = "mixlab"
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.model_dim)
        self.char_table = None
        self.char_proj = None
        self.char_scale = None
        self.bigram_table = None
        self.bigram_proj = None
        self.bigram_scale = None
        self.trigram_table = None
        self.trigram_proj = None
        self.trigram_scale = None
        if int(getattr(config, "char_vocab_size", 0) or 0) > 0:
            char_dim = int(getattr(config, "char_dim", 0) or config.model_dim)
            self.char_table = nn.Embedding(int(config.char_vocab_size), char_dim)
            if char_dim != config.model_dim:
                self.char_proj = MixlabLinear(char_dim, config.model_dim)
            self.char_scale = nn.Parameter(torch.ones(1))
            self.register_buffer("char_lookup", load_char_lookup(config), persistent=False)
        if int(getattr(config, "bigram_vocab_size", 0) or 0) > 0:
            bigram_dim = int(getattr(config, "bigram_dim", 0) or config.model_dim)
            self.bigram_table = nn.Embedding(int(config.bigram_vocab_size), bigram_dim)
            if bigram_dim != config.model_dim:
                self.bigram_proj = MixlabLinear(bigram_dim, config.model_dim)
            self.bigram_scale = nn.Parameter(torch.ones(1))
        if int(getattr(config, "trigram_vocab_size", 0) or 0) > 0:
            trigram_dim = int(getattr(config, "trigram_dim", 0) or config.model_dim)
            self.trigram_table = nn.Embedding(int(config.trigram_vocab_size), trigram_dim)
            if trigram_dim != config.model_dim:
                self.trigram_proj = MixlabLinear(trigram_dim, config.model_dim)
            self.trigram_scale = nn.Parameter(torch.ones(1))
        modules = []
        for block in config.blocks:
            block_type = block.get("type")
            if block_type == "plain":
                modules.append(MixlabPlainBlock(config, block))
            elif block_type == "swiglu":
                modules.append(MixlabSwiGLUBlock(config, "sigmoid"))
            elif block_type == "geglu":
                modules.append(MixlabSwiGLUBlock(config, "gelu"))
            elif block_type == "mlp":
                modules.append(MixlabMLPBlock(config, block))
            elif block_type == "moe":
                modules.append(MixlabMoEBlock(config, block))
            else:
                raise ValueError(f"unsupported exported Mixlab block type {block_type!r}")
        self.blocks = nn.ModuleList(modules)
        self.final_norm = MixlabRMSNorm(config.model_dim)
        self.lm_head_weight = nn.Parameter(torch.empty(config.model_dim, config.vocab_size))
        nn.init.xavier_uniform_(self.lm_head_weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _bigram_ids(self, input_ids):
        vocab = int(self.config.bigram_vocab_size)
        # Match Mixlab native feature construction, which hashes the flattened
        # token buffer and intentionally carries the predecessor across rows.
        flat = input_ids.reshape(-1).to(torch.long)
        ids = torch.empty_like(flat)
        if flat.numel() == 0:
            return ids.view_as(input_ids)
        modulus = vocab - 1
        ids[0] = modulus
        if flat.numel() > 1:
            ids[1:] = ((36313 * flat[1:]) ^ (27191 * flat[:-1])) % modulus
        return ids.view_as(input_ids)

    def _trigram_ids(self, input_ids):
        vocab = int(self.config.trigram_vocab_size)
        modulus = vocab - 1
        out = torch.zeros_like(input_ids, dtype=torch.long)
        if input_ids.shape[1] <= 2:
            return out
        t0 = input_ids[:, :-2].to(torch.long)
        t1 = input_ids[:, 1:-1].to(torch.long)
        t2 = input_ids[:, 2:].to(torch.long)
        h = ((t0 * vocab + t1) * vocab + t2) % modulus
        out[:, 2:] = h + 1
        return out

    def _embed_features(self, input_ids):
        x = self.embed_tokens(input_ids)
        if self.char_table is not None:
            char_ids = self.char_lookup[input_ids]
            char_emb = self.char_table(char_ids)
            char_emb = char_emb * char_ids.ne(0).unsqueeze(-1)
            char_state = char_emb.sum(dim=-2)
            if self.char_proj is not None:
                char_state = self.char_proj(char_state)
            x = x + char_state * self.char_scale
        if self.bigram_table is not None:
            ids = self._bigram_ids(input_ids)
            state = self.bigram_table(ids)
            if self.bigram_proj is not None:
                state = self.bigram_proj(state)
            x = x + state * self.bigram_scale
        if self.trigram_table is not None:
            ids = self._trigram_ids(input_ids)
            state = self.trigram_table(ids)
            if self.trigram_proj is not None:
                state = self.trigram_proj(state)
            x = x + state * self.trigram_scale
        return x

    def forward(self, input_ids=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        x = self._embed_features(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = torch.matmul(x, self.lm_head_weight)
        if getattr(self.config, "logit_softcap", 0.0):
            cap = float(self.config.logit_softcap)
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)
