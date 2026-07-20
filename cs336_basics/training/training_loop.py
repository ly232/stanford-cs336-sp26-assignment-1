'''Main script to run training loop.

uv run python cs336_basics/training/training_loop.py
'''

import tqdm
import torch
import yaml

from jaxtyping import Float

from cs336_basics.data import checkpointing
from cs336_basics.data import data_loader
from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.nn_utils import utils
from cs336_basics.training.adamw_optimizer import AdamW
from cs336_basics.tokenizer.bpe import BytePairEncoder
from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer
from cs336_basics.tokenizer.tokenizer import Tokenizer

# Detect hardware device.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def main():
    # Parse job configs to get hyperparameters.
    num_steps = ...
    batch_size = ...
    vocab_size = ...
    context_length = ...
    d_model = ...
    num_layers = ...
    num_heads = ...
    d_ff = ...
    rope_theta = ...
    special_tokens = ...
    checkpointing_interval = ...
    output_file = ...
    # AdamW optimizer gradient stability parameters
    max_l2_norm = ...
    max_learning_rate = ...
    min_learning_rate = ...
    warmup_iters = ...

    #
    # Data loading.
    #
    training_text = ...  # Load from training text file.

    #
    # Tokenization.
    #
    pretokenizer = SpecialTokenAwarePretokenizer(special_tokens)
    pretokens = pretokenizer.pretokenize(training_text)
    bpe = BytePairEncoder(pretokens, vocab_size, special_tokens)
    vocabs, merges = bpe.train()
    tokenizer = Tokenizer(vocabs, merges, special_tokens)
    dataset = tokenizer.encode(training_text)

    #
    # Training.
    #
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
    )
    losses = []
    for it in tqdm.tqdm(range(num_steps)):
        inputs, targets = data_loader.get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        optimizer.zero_grad()
        predictions: Float[
            torch.Tensor,
            'batch seq vocab',
        ] = model(inputs)
        loss = utils.cross_entropy(
            logits=predictions,
            targets=targets,
        )
        losses.append(loss.cpu().item())
        loss.backward()
        # Apply gradient clipping and learning rate scheduling for stability.
        utils.clip_gradient(model.parameters(), max_l2_norm=max_l2_norm)
        lr = utils.get_lr_cosine_schedule(
            it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=num_steps)
        for group in optimizer.param_groups:
            group['alpha'] = lr
        optimizer.step()

        # Checkpointing
        if it % checkpointing_interval == 0:
            checkpointing.save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it,
                out=output_file,
            )

    # Save one last time.
    checkpointing.save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=it,
        out=output_file,
    )

if __name__ == '__main__':
    main()
