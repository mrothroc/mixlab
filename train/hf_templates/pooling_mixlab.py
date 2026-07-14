import torch


def token_validity_mask(hidden, attention_mask):
    if hidden.ndim != 3:
        raise ValueError(
            "sequence pooling requires hidden states shaped [batch, sequence, hidden]"
        )
    batch_size, seq_len = hidden.shape[:2]
    if attention_mask is None:
        if batch_size > 1:
            raise ValueError(
                "sequence pooling requires a 2D attention_mask when batch_size > 1"
            )
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden.device)
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, seq_len):
        raise ValueError(
            "sequence pooling requires attention_mask shaped [batch, sequence]; "
            f"got {tuple(attention_mask.shape)} for hidden states {tuple(hidden.shape)}"
        )
    mask = attention_mask.to(device=hidden.device).ne(0)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("sequence pooling received a row with no real tokens")
    return mask


def last_real_index(attention_mask):
    if attention_mask.ndim != 2:
        raise ValueError("last_real_index requires a 2D attention_mask")
    mask = attention_mask.ne(0)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("last_real_index received a row with no real tokens")
    seq_len = mask.shape[-1]
    return seq_len - 1 - mask.flip(-1).to(torch.long).argmax(dim=-1)


def pool_last(hidden, attention_mask=None):
    mask = token_validity_mask(hidden, attention_mask)
    indices = last_real_index(mask)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


def pool_mean(hidden, attention_mask=None):
    mask = token_validity_mask(hidden, attention_mask)
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


def pool_sequence(hidden, attention_mask, mode):
    mode = str(mode or "").strip().lower()
    if mode == "last":
        return pool_last(hidden, attention_mask)
    if mode == "mean":
        return pool_mean(hidden, attention_mask)
    raise ValueError(
        "sequence_classification_pooling must be 'last' or 'mean'; "
        "pass an explicit value when the exported backbone is ambiguous"
    )
