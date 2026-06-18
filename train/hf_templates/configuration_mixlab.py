from transformers import PretrainedConfig


class MixlabConfig(PretrainedConfig):
    model_type = "mixlab"

    def __init__(
        self,
        model_dim=0,
        vocab_size=0,
        seq_len=0,
        mlp_mult=2.67,
        norm_type="rmsnorm",
        norm_eps=1e-5,
        norm_affine=True,
        norm_placement="pre",
        ffn_internal_norm=False,
        blocks=None,
        masked_blocks=None,
        logit_softcap=0.0,
        mlm_head="linear",
        hidden_dropout=0.0,
        char_vocab_size=0,
        char_dim=0,
        char_max_per_token=0,
        char_features_file="",
        bigram_vocab_size=0,
        bigram_dim=0,
        trigram_vocab_size=0,
        trigram_dim=0,
        **kwargs,
    ):
        # Exported checkpoints always materialize a separate lm_head weight, so
        # the HF model must not tie embeddings to the head unless explicitly told.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.hidden_size = model_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_position_embeddings = seq_len
        self.mlp_mult = mlp_mult
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.norm_placement = norm_placement
        self.ffn_internal_norm = ffn_internal_norm
        self.blocks = blocks or []
        self.masked_blocks = masked_blocks or []
        self.logit_softcap = logit_softcap
        self.mlm_head = mlm_head
        self.hidden_dropout = hidden_dropout
        self.char_vocab_size = char_vocab_size
        self.char_dim = char_dim
        self.char_max_per_token = char_max_per_token
        self.char_features_file = char_features_file
        self.bigram_vocab_size = bigram_vocab_size
        self.bigram_dim = bigram_dim
        self.trigram_vocab_size = trigram_vocab_size
        self.trigram_dim = trigram_dim
