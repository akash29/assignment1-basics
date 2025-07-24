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


def count_pairs(chunk_ids):
    freq_map = collections.defaultdict(int)
    for u,v in zip(chunk_ids[:-1], chunk_ids[1:]):
        freq_map[(u, v)] += 1
    return freq_map


def apply_merge(chunk_ids, merge_pair, new_idx):
    i = 0
    out = []
    while i < len(chunk_ids):
        if i < len(chunk_ids) -1 and (chunk_ids[i], chunk_ids[i+1]) == merge_pair:
            out.append(new_idx)
            i+=2
        else:
            out.append(chunk_ids[i])
            i+=1
    return out


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    cpu_count = mp.cpu_count()
    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = bytes([i])
    
    for token in special_tokens:
        idx = len(vocab)
        vocab[idx] = token.encode('utf-8')

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, cpu_count, special_tokens[0].encode('utf-8'))

    all_doc_ids = []
    all_freq_counts = collections.defaultdict(int)
    with mp.Pool(processes=cpu_count) as pool:
        results = pool.starmap(
            get_chunk_ids,
            [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        for res in results:
            all_doc_ids.extend(res)

        while len(vocab) < vocab_size:
            freq_count_res = pool.starmap(count_pairs, [(chunk_ids,) for chunk_ids in all_doc_ids])
            all_freq_counts.clear()
            for freq_count in freq_count_res:
                for k, v in freq_count.items():
                    all_freq_counts[k] += v

            if not all_freq_counts:
                print("No more pairs to merge.")
                break
            max_frequency_pair = max(all_freq_counts.items(), key=lambda x: x[1])[0]

            idx = len(vocab)
            vocab[idx] = vocab[max_frequency_pair[0]] + vocab[max_frequency_pair[1]]
            merges.append((vocab[max_frequency_pair[0]], vocab[max_frequency_pair[1]]))

            id_count_res = pool.starmap(apply_merge,[(chunk_ids, max_frequency_pair, idx) for chunk_ids in all_doc_ids])
            all_doc_ids = id_count_res[:]
            
    return vocab, merges    

if __name__ == "__main__":
    input_path = "data/tiny_test.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(vocab)
    print(merges)
    
