import torch


def _require_nonempty_rows(mask, message):
    valid_rows = mask.any(dim=-1)
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        torch._assert_async(torch.all(valid_rows), message)
        return
    if not bool(torch.all(valid_rows)):
        raise ValueError(message)


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
    _require_nonempty_rows(mask, "sequence pooling received a row with no real tokens")
    return mask


def _last_index_of_valid_mask(mask):
    # Padding-side agnostic: works for right padding (final real token before
    # trailing pads) and left padding alike, with no branch on padding side.
    seq_len = mask.shape[-1]
    return seq_len - 1 - mask.flip(-1).to(torch.long).argmax(dim=-1)


def last_real_index(attention_mask):
    if attention_mask.ndim != 2:
        raise ValueError("last_real_index requires a 2D attention_mask")
    mask = attention_mask.ne(0)
    _require_nonempty_rows(mask, "last_real_index received a row with no real tokens")
    return _last_index_of_valid_mask(mask)


def pool_last(hidden, attention_mask=None):
    # token_validity_mask already asserts non-empty rows, so index the validated
    # mask directly rather than re-running last_real_index's guard.
    mask = token_validity_mask(hidden, attention_mask)
    indices = _last_index_of_valid_mask(mask)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


def pool_mean(hidden, attention_mask=None):
    mask = token_validity_mask(hidden, attention_mask)
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


def cls_layout(x, attention_mask, position, configured_length=None):
    # x supplies only the raw sequence shape/device. Indices address the
    # prefixed [CLS, content...] tensor, identically for embeddings and masks.
    batch, length = x.shape[:2]
    if attention_mask is None:
        mask = torch.ones((batch, length), dtype=torch.long, device=x.device)
    else:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, length):
            raise ValueError("CLS attention_mask must have input shape [batch, sequence]")
        mask = attention_mask.to(device=x.device)
    indices = torch.arange(length + 1, device=x.device).expand(batch, -1)
    positions = torch.zeros(batch, dtype=torch.long, device=x.device)
    if position != "head":
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("CLS placement requires a binary attention_mask")
        lengths = mask.long().sum(dim=1)
        prefix = torch.arange(length, device=x.device)[None, :] < lengths[:, None]
        if not bool(torch.all(mask.bool() == prefix)) or not bool(torch.all(lengths > 0)):
            raise ValueError("CLS placement requires non-empty right-padded valid prefixes")
        if position == "middle":
            if length != configured_length or not bool(torch.all(lengths == length)):
                raise ValueError("cls_position=middle requires fixed full-length records at seq_len")
            positions = lengths // 2
        elif position == "tail":
            positions = lengths
        else:
            raise ValueError("cls_position must be head, middle, or tail")
        indices = torch.where(indices < positions[:, None], indices + 1, indices)
        is_cls = torch.arange(length + 1, device=x.device)[None, :] == positions[:, None]
        indices = torch.where(is_cls, 0, indices)
    cls_mask = torch.ones((batch, 1), dtype=mask.dtype, device=x.device)
    expanded = torch.cat((cls_mask, mask), dim=1)
    return indices, positions, expanded.gather(1, indices)


def insert_cls(x, token, attention_mask, position, configured_length):
    prefixed = torch.cat((token.unsqueeze(0).expand(x.shape[0], -1, -1), x), dim=1)
    if position == "head":
        return prefixed
    indices, _, _ = cls_layout(x, attention_mask, position, configured_length)
    return prefixed.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def pool_sequence(hidden, attention_mask, mode, cls_position="head", configured_length=None):
    mode = str(mode or "").strip().lower()
    if mode == "cls":
        if cls_position == "head":
            return hidden[:, 0]
        _, positions, _ = cls_layout(hidden[:, 1:], attention_mask, cls_position, configured_length)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
    if mode == "last":
        return pool_last(hidden, attention_mask)
    if mode == "mean":
        return pool_mean(hidden, attention_mask)
    raise ValueError(
        "sequence_classification_pooling must be 'last', 'mean', or 'cls'; "
        "pass an explicit value when the exported backbone is ambiguous"
    )
