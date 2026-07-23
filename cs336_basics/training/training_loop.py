'''Main script to run training loop.

uv run python cs336_basics/training/training_loop.py --corpus tinystories_sample_5M
'''

import argparse
import functools
import numpy as np
import tqdm
import torch
import yaml

from numpy import typing as npt
from jaxtyping import Float

from cs336_basics.data import checkpointing
from cs336_basics.data import data_loader
from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.nn_utils import utils
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.training.adamw_optimizer import AdamW
from cs336_basics.training.input_file_chunker import read_chunk, split_and_process
from cs336_basics.tokenizer.bpe import BytePairEncoder
from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer
from cs336_basics.tokenizer.tokenizer import Tokenizer

# Detect hardware device.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def tokenizer_encode_chunk(
    input_file: str,
    vocabs_file: str,
    merges_file: str,
    special_tokens: list[str],
    start_end: tuple[int, int],
) -> npt.NDArray[np.int32]:
    '''Tokenizes a chunk (start_end) from input_file.'''
    tokenizer = Tokenizer.from_files(
        vocabs_file,
        merges_file,
        special_tokens,
    )
    training_text = read_chunk(input_file, start_end)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocabs_file,
        merges_filepath=merges_file,
        special_tokens=special_tokens,
    )
    vocab_ids = tokenizer.encode(training_text)
    return np.array(vocab_ids, dtype=np.int32)


def main():
    # Parse job configs to get hyperparameters.
    with open('./cs336_basics/training/training_loop.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    config = configs[args.corpus]
    # Training configs.
    input_file = config['input_file']
    pretokens_vocab_ids_file = config['pretokens_vocab_ids_file']
    num_steps = config.get('num_steps', 100)
    checkpointing_interval = config.get(
        'checkpointing_interval', num_steps // 10)
    output_file = config['output_file']
    losses_file = config['losses_file']
    # Model configs.
    batch_size = config.get('batch_size', 32)
    vocab_size = config.get('vocab_size', 10000)
    context_length = config.get('context_length', 512)
    d_model = config.get('d_model', 32)
    num_layers = config.get('num_layers', 8)
    num_heads = config.get('num_heads', 4)
    d_ff = config.get('d_ff', 64)
    rope_theta = config.get('rope_theta', 10000)
    # Tokenizer configs.
    vocabs_file = config['vocabs_file']
    merges_file = config['merges_file']
    special_tokens = config.get('special_tokens', ['<|endoftext|>'])
    # Optimizer configs.
    max_l2_norm = config.get('max_l2_norm', 1.0)
    max_learning_rate = config.get('max_learning_rate', 1e-4)
    min_learning_rate = config.get('min_learning_rate', 1e-5)
    warmup_iters = config.get('warmup_iters', int(num_steps * 0.1))

    # Split files into chunks to process independently.
    pretokens_vocab_ids: npt.NDArray[np.int32] = split_and_process(
        input_file=input_file,
        special_tokens=special_tokens,
        map_fn=functools.partial(
            tokenizer_encode_chunk,
            input_file,
            vocabs_file,
            merges_file,
            special_tokens,
        ),
        reduce_fn=np.concatenate,
    )
    np.save(pretokens_vocab_ids_file, pretokens_vocab_ids)
    print(f'Saved pretokens vocab ids to {pretokens_vocab_ids_file}')

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
            dataset=pretokens_vocab_ids,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )  # batch, seq_len
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
            print(f'Saving checkpoint at iteration {it} to file {output_file}')
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
    print(f'Saved final checkpoint into {output_file}')

    np.save(losses_file, np.array(losses, dtype=np.float32))
    print(f'Saved losses file to {losses_file}')

if __name__ == '__main__':
    # Commandline args parsing.
    parser = argparse.ArgumentParser(description='Main training loop.')
    parser.add_argument('--corpus', type=str, default='tinystories_sample_5M', 
                        help='training corpus, e.g. tinystories_sample_5M')
    args = parser.parse_args()

    main()
