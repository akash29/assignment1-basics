import regex as re
import multiprocessing as mp
from pretokenization_example import *
import collections
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


vocab = {}
reverse_vocab = {}
frequency_map = collections.defaultdict(int)
merges_map = {}
for i in range(256):
    vocab[i] = bytes([i])
    reverse_vocab[bytes([i])] = i


def process_chunk(input_path, special_tokens, start, end, vocab, reverse_vocab):
    pattern = "|".join(map(re.escape, special_tokens))
    frequency = collections.defaultdict(int)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunks = f.read(end - start).decode("utf-8", errors="ignore") 
        # pre-tokenization
        chunks = re.split(pattern, chunks)
        for chunk in chunks:
            words = list(map(lambda x: x.group(0),re.finditer(PAT.encode('utf-8'), chunk.encode('utf-8'))))
            for word in words:
                if len(word) < 2:
                    continue
                for pair in zip(word, word[1:]):
                    frequency[pair] += 1
    return frequency
                

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    merges = []

    chunk_count = mp.cpu_count()
    print(f"Using {chunk_count} processes for BPE training.")

    idx = len(vocab)
    for token in special_tokens:
        enc_token = token.encode("utf-8")
        vocab[idx] = enc_token
        reverse_vocab[enc_token] = idx
        idx+=1


    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_count, "<|endoftext|>".encode("utf-8"))

    max_count = float('-inf')
    max_pair = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args = (input_path, special_tokens, start, end, vocab, reverse_vocab)
        with mp.Pool(processes=chunk_count) as pool:
            for res in pool.starmap(process_chunk, [args]):
                for pair, count in res.items():
                    frequency_map[pair]+=count
                    if frequency_map[pair] > max_count:
                        max_count = frequency_map[pair]
                        max_pair = pair
    
    idx = len(vocab)
    vocab[idx] = bytes(max_pair)
    
    merges_map[max_pair] = idx
    merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
    print (merges_map)
    print(vocab)
    print(merges)
    

            


        
if __name__ == "__main__":
    train_bpe("data/tiny_test.txt", 1000, ["<|endoftext|>"])




