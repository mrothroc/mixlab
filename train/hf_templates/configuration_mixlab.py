from transformers import PretrainedConfig


class MixlabConfig(PretrainedConfig):
    model_type = "mixlab"

    def __init__(
        self,
        model_dim=0,
        vocab_size=0,
        seq_len=0,
        mlp_mult=2.67,
        blocks=None,
        logit_softcap=0.0,
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
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.hidden_size = model_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_position_embeddings = seq_len
        self.mlp_mult = mlp_mult
        self.blocks = blocks or []
        self.logit_softcap = logit_softcap
        self.char_vocab_size = char_vocab_size
        self.char_dim = char_dim
        self.char_max_per_token = char_max_per_token
        self.char_features_file = char_features_file
        self.bigram_vocab_size = bigram_vocab_size
        self.bigram_dim = bigram_dim
        self.trigram_vocab_size = trigram_vocab_size
        self.trigram_dim = trigram_dim
