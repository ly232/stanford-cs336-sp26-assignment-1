'''Trains the byte pair encoder.

uv run python -m cProfile -s tottime cs336_basics/training/train_bpe.py --corpus tinystories_sample_5M | head -n 100
uv run python -m cProfile -s tottime cs336_basics/training/train_bpe.py --corpus openwebtext | head -n 100
'''

import argparse
import collections
import pickle

import yaml

# from cs336_basics.tokenizer.bpe import BytePairEncoder
from cs336_basics.tokenizer.bpe_v2 import BytePairEncoder
from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer
from cs336_basics.tokenizer.tokenizer import Tokenizer

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

    print('Loading training text and initiating BPE with pretokens.')
    bpe = BytePairEncoder(vocab_size, special_tokens)
    pretokenizer = SpecialTokenAwarePretokenizer(special_tokens)

    # Load data file.
    with open(input_file, 'r') as f:
        training_text = f.read()
    pretokens = pretokenizer.pretokenize(training_text)
    bpe.update_pretokens(collections.Counter(pretokens))

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
                        help='training corpus, e.g. tinystories_sample_5M, or openwebtext')
    args = parser.parse_args()

    # Main.
    main()
