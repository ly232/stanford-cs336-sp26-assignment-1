from dataclasses import dataclass

from cs336_basics.model.transformer_lm import TransformerLanguageModel

@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    def __post_init__(self):
        self.model = TransformerLanguageModel(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
        )

    def count_params(self):
        return {
            'all_params': sum(
                p.numel()
                for p in self.model.parameters()
            ),
            'trainable_params': sum(
                p.numel()
                for p in self.model.parameters()
                if p.requires_grad
            )
        }
