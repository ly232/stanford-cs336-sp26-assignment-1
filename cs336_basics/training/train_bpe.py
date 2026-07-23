'''Trains the byte pair encoder.

uv run python cs336_basics/training/train_bpe.py --corpus tinystories_sample_5M
uv run python -m cProfile -s tottime cs336_basics/training/train_bpe.py --corpus tinystories_sample_5M | head -n 100

uv run python cs336_basics/training/train_bpe.py --corpus openwebtext_validation
uv run python -m cProfile -s tottime cs336_basics/training/train_bpe.py --corpus openwebtext_validation | head -n 100

uv run python cs336_basics/training/train_bpe.py --corpus openwebtext_train
uv run python -m cProfile -s tottime cs336_basics/training/train_bpe.py --corpus openwebtext_train | head -n 100
'''

import argparse
import collections
import functools
import itertools
import multiprocessing
import numpy as np
import os
import pickle

from cs336_basics.training.input_file_chunker import read_chunk, split_and_process
from cs336_basics.tokenizer.bpe_v2 import BytePairEncoder
from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer

import yaml

def pretokenize_chunk(
    input_file: str,
    special_tokens: list[str],
    start_end: tuple[int, int])\
    -> collections.Counter:
    '''Pretokenizes a single file chunk file.

    Note this method is intended to be used within multiprocessing pool. As
    such, it needs to be top-level function.

    Args:
      input_file: input file.
      special_tokens: special tokens, e.g. ['<|endoftext|>']
      start_end: tuple of start and end byte positions of the input file.

    Returns a counter from pretoken to count of the file chunk.
    '''
    training_text = read_chunk(input_file=input_file, start_end=start_end)
    pretokenizer = SpecialTokenAwarePretokenizer(special_tokens)
    return collections.Counter(pretokenizer.pretokenize(training_text))

def generate_pretokens_counter(
    input_file: str,
    special_tokens: list[str],
) -> collections.Counter:
    map_fn = functools.partial(
        pretokenize_chunk,
        input_file,
        special_tokens,
    )
    def reduce_fn(child_counters):
        # Aggregate results from individual chunk's pretoken counters.
        pretokens_counter = collections.Counter()
        for counter in child_counters:
            pretokens_counter += counter
        return pretokens_counter
    return split_and_process(
        input_file=input_file,
        special_tokens=special_tokens,
        map_fn=map_fn,
        reduce_fn=reduce_fn,
    )
    

def main():
    # Parse job configs to get hyperparameters.
    with open('./cs336_basics/training/train_bpe.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    config = configs[args.corpus]
    input_file = config['input_file']
    special_tokens = config['special_tokens']
    vocab_size = config['vocab_size']
    output_bpe_vocabs_file = config['output_bpe_vocabs_file']
    output_bpe_merges_file = config['output_bpe_merges_file']
    output_pretokens_counter_file = config[
        'output_pretokens_counter_file']
    pretokenize_with_multiprocessing = config.get(
        'pretokenize_with_multiprocessing', False)

    print('Loading training text and initiating BPE with pretokens.')
    bpe = BytePairEncoder(vocab_size, special_tokens)

    # Load data file.
    print('Pretokenizing training text...')
    if pretokenize_with_multiprocessing:
        pretokens_counter = generate_pretokens_counter(
            input_file=input_file,
            special_tokens=special_tokens,
        )
    else:
        with open(input_file, 'r') as f:
            training_text = f.read()
        pretokenizer = SpecialTokenAwarePretokenizer(special_tokens)
        pretokens = pretokenizer.pretokenize(training_text)
        pretokens_counter = collections.Counter(pretokens)
    print('Saving pretokens_counter...')
    with open(output_pretokens_counter_file, 'wb') as f:
        pickle.dump(pretokens_counter, f)
    print('Updating pretokens into BPE...')
    bpe.update_pretokens(pretokens_counter)

    # Start training.
    print('Training BPE...')
    vocabs, merges = bpe.train()
    print('Saving trained vocabs and merges...')
    with open(output_bpe_vocabs_file, 'wb') as f:
        pickle.dump(vocabs, f)
    with open(output_bpe_merges_file, 'wb') as f:
        pickle.dump(merges, f)
    print('ALL DONE!')

if __name__ == '__main__':
    # Commandline args parsing.
    parser = argparse.ArgumentParser(description='BPE trainer.')
    parser.add_argument('--corpus', type=str, default='tinystories_sample_5M', 
                        help='training corpus, e.g. tinystories_sample_5M')
    args = parser.parse_args()

    # Main.
    main()
