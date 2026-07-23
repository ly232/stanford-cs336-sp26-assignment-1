'''Util to chunk input files.

This is optional for small files, but for large files such as OpenWebText,
where the training input can span over 11GiB, we cannot load everything in
one shot, thus we need to split to chunks, then sequentialize the work.
'''

from typing import Any, Callable, Iterable

import itertools
import multiprocessing

from cs336_basics.pretokenization_example import find_chunk_boundaries

def read_chunk(
    input_file: str,
    start_end: tuple[int, int],
) -> str:
    '''Reads a chunk within given input file.'''
    start, end = start_end
    with open(input_file, 'rb') as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")

def split_and_process(
    input_file: str,
    special_tokens: list[str],
    map_fn: Callable[..., Any],
    reduce_fn: Callable[..., Any] = lambda x: x,
    num_workers: int = 2,
    num_chunks: int = 10,
) -> Any:
    '''Splits the input file into chunks to process
    
    Args:
      input_file: input file to split and process.
      special_tokens: list of special tokens, only supports 1 token.
      map_fn: custom mapper callbcak; see
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map
        for details.
      reduce_fn: custom reducer callback.
      num_workers: number of parallel workers to process.
        ATTN: be conservative at this, as large numbers may halt the OS.
      num_chunks: number of chunks to split the input file.

    Returns the result from reduce_fn.
    '''
    assert len(special_tokens) == 1, 'Only supports 1 special token.'
        
    with open(input_file, 'rb') as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, special_tokens[0].encode())

    with multiprocessing.Pool(num_workers) as pool:
        # Assign file split indexes from the iterable into the lambda.
        results = pool.map(
            map_fn, itertools.pairwise(boundaries))

    return reduce_fn(results)
