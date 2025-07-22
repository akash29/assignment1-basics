import regex as re
import multiprocessing as mp
from pretokenization_example import *
import collections
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_chunk_ids(input_path, start, end, special_tokens):
    pattern = "|".join(map(re.escape, special_tokens))
    chunk_ids = []

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end-start)
        split_chunks = re.split(pattern.encode('utf-8'), chunk)
        for split_chunk in split_chunks:
            words = [i.group(0) for i in re.finditer(PAT.encode('utf-8'), split_chunk)]
            doc_ids = [byte for w in words for byte in w]
            if doc_ids:
                chunk_ids.append(doc_ids)

    return chunk_ids


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    cpu_count = mp.cpu_count()
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    for token in special_tokens:
        idx = len(vocab)
        vocab[idx] = token.encode('utf-8')

    print(vocab)

    all_doc_ids = []
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, cpu_count, special_tokens[0].encode('utf-8'))

        with mp.Pool(processes=cpu_count) as pool:
            results = pool.starmap(
                get_chunk_ids,
                [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
            )
            for res in results:
                all_doc_ids.extend(res)
    
    print(all_doc_ids)
    
    

if __name__ == "__main__":
    input_path = "data/tiny_test.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    train_bpe(input_path, vocab_size, special_tokens)
    
