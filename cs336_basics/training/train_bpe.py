'''Trains the byte pair encoder.

uv run python cs336_basics/training/train_bpe.py
'''

import pickle
import yaml

from cs336_basics.tokenizer.bpe import BytePairEncoder
from cs336_basics.tokenizer.pretokenizer import SpecialTokenAwarePretokenizer
from cs336_basics.tokenizer.tokenizer import Tokenizer

def main():
    # Parse job configs to get hyperparameters.
    with open('./cs336_basics/training/train_bpe.yaml', 'r') as f:
        config = yaml.safe_load(f)
    input_file = config['input_file']
    special_tokens = config['special_tokens']
    vocab_size = config['vocab_size']
    output_bpe_vocabs_file = config['output_bpe_vocabs_file']
    output_bpe_merges_file = config['output_bpe_merges_file']

    # Load training text.
    with open(input_file, 'r') as f:
        training_text = f.read()
    print(f'Loaded BPE training file {input_file}')

    # Train BPE.
    pretokenizer = SpecialTokenAwarePretokenizer(special_tokens)
    pretokens = pretokenizer.pretokenize(training_text)
    bpe = BytePairEncoder(pretokens, vocab_size, special_tokens)
    print('Training BPE...')
    vocabs, merges = bpe.train()
    print('Saving trained vocabs and merges...')
    with open(output_bpe_vocabs_file, 'wb') as f:
        pickle.dump(vocabs, f)
    with open(output_bpe_merges_file, 'wb') as f:
        pickle.dump(merges, f)
    print('ALL DONE!')

if __name__ == '__main__':
    main()
