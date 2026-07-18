from collections import Counter
from dataclasses import dataclass

from cs336_basics.analysis.transformer_config import TransformerConfig

@dataclass
class FlopsCountingNode:
    name: str
    children: list['FlopsCountingNode'] | None = None
    flops: int = 0
    
    def count(self):
        return self.flops + (
            sum(c.count() for c in self.children)
            if self.children else 0
        )
    
    def breakdown(self):
        '''Generates flops breakdown for this node's subtree.'''
        result = {
            self.name: self.flops
        }
        if self.children:
            for i, child in enumerate(self.children):
                result[f'{child.name}[{i}]'] = child.count()
        return result

    @classmethod
    def create_input_embedding(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        # We use torch.gather without materializing token indices
        # to onehot encoding, so input embedding layer becomes a
        # lookup with 0 FLOPs.
        flops = 0
        return cls(
            name='InputEmbedding',
            flops=flops,
        )
    
    @classmethod
    def create_rms_norm(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        flops = (
            2 
            * config.context_length 
            * config.d_model
        )
        return cls(
            name='RmsNorm',
            flops=flops,
        )

    @classmethod
    def create_rope(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        # input x is (... seq dk), rot mat is (dk dk)
        d_k = config.d_model // config.num_heads
        flops = 2 * config.context_length * d_k
        return cls(
            name='RotaryPositionalEmbedding',
            flops=flops,
        )
    
    @classmethod
    def create_single_head_linear_layer(
        cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        # Applicable to single head q, k, and v layers.
        # While attention don't require same dimensions for
        # k, q, v, for transformer lm we often make them
        # align for simplicity and hardware efficiency.
        d_k = config.d_model // config.num_heads
        flops = 2 * config.context_length * config.d_model * d_k
        return cls(
            name='SingleHeadLinear',
            flops = flops,
        )
    
    @classmethod
    def create_mha_output_layer(
        cls, config: TransformerConfig) \
        -> 'FlopsCountingNode':
        return cls(
            name='MhaOutput',
            flops=(
                2
                * config.context_length
                * config.d_model
                * config.d_model),
        )

    @classmethod
    def create_mha(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        def single_head_scaled_dot_product_attention_flops():
            # While scaled dot product attention doesn't
            # require same dimensions for Q, K, and V
            # matrices, in transformer LM they often match.
            queries = config.context_length
            keys = config.context_length
            d_k = config.d_model // config.num_heads
            d_v = d_k

            qk_flops = 2 * queries * keys * d_k
            softmax_v_flops = 2 * queries * keys * d_v

            return qk_flops + softmax_v_flops
        
        attention_flops = single_head_scaled_dot_product_attention_flops()\
            * config.num_heads
        return cls(
            name='MultiHeadAttention',
            flops=attention_flops,
            children=[
                cls.create_rope(config),
            ] + [
                cls.create_single_head_linear_layer(config)
                # multiply by 3 to account for q k v
                for _ in range(config.num_heads * 3)
            ] + [
                cls.create_mha_output_layer(config)
            ],
        )
    
    @classmethod
    def create_positionwise_ffn_w1_layer(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='SiluW1',
            # input.shape = (batch seq d_model)
            # w1.shape = (batch d_model d_ff)
            flops=2 * config.d_model * config.d_ff * config.context_length,
        )
    
    @classmethod
    def create_positionwise_ffn_w2_layer(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='SiluW2',
            # input.shape = (batch seq d_ff)
            # w2.shape = (batch d_ff d_model)
            flops=2 * config.d_model * config.d_ff * config.context_length,
        )
    
    @classmethod
    def create_positionwise_ffn_w3_layer(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='SiluW3',
            # input.shape = (batch seq d_model)
            # w3.shape = (batch d_model d_ff)
            flops=2 * config.d_model * config.d_ff * config.context_length,
        )

    @classmethod
    def create_swiglu_ffn(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        # NOTE: this is the MOST EXPENSIVE layer flops-wise.
        return cls(
            name='PositionwiseFeedforward',
            children=[
                cls.create_positionwise_ffn_w1_layer(config),
                cls.create_positionwise_ffn_w2_layer(config),
                cls.create_positionwise_ffn_w3_layer(config),
            ]
        )

    @classmethod
    def create_transformer_block(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='TransformerBlock',
            children=[
                cls.create_rms_norm(config),
                cls.create_mha(config),
                cls.create_rms_norm(config),
                cls.create_swiglu_ffn(config),
            ]
        )

    @classmethod
    def create_output_embedding(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='OutputEmbedding',
            flops=(
                2
                * config.context_length
                * config.vocab_size
                * config.d_model),
        )

    @classmethod
    def create_transformer_llm(cls, config: TransformerConfig)\
        -> 'FlopsCountingNode':
        return cls(
            name='TransformerLanguageModel',
            children=[
                cls.create_input_embedding(config),
            ] + [
                cls.create_transformer_block(config)
                for _ in range(config.num_layers)
            ] + [
                cls.create_rms_norm(config),
                cls.create_output_embedding(config),
            ],
        )

def create_llm_accounting_node(config: TransformerConfig)\
    -> FlopsCountingNode:
    '''Factory method to create root accounting node.'''
    return FlopsCountingNode.create_transformer_llm(config)
