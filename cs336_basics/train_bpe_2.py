import regex as re
import multiprocessing as mp
from pretokenization_example import *
import collections
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

all_freq_counts = collections.defaultdict(int)

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


def count_pairs(doc_id, chunk_ids):
    freq_map = collections.defaultdict(int)
    pair_locations = collections.defaultdict(list)
    for pos_idx, (u,v) in enumerate(zip(chunk_ids[:-1], chunk_ids[1:])):
        freq_map[(u, v)] += 1
        pair_locations[(u, v)].append((doc_id, pos_idx))
    return freq_map, pair_locations


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

def update_counts(pair_locations, chunks, merge_pair, new_token):
    pair_location_list = pair_locations[merge_pair].copy()
    for cur_doc_idx, pos_idx in pair_location_list:
        chunk_ids = chunks[cur_doc_idx]
        preceeding_pos = pos_idx - 1
        following_pos = pos_idx + 2
        if preceeding_pos < 0 or following_pos >= len(chunk_ids):
            continue
    
        preceeding_pair = (chunk_ids[preceeding_pos], merge_pair[0]) if preceeding_pos >= 0 else None
        new_pair_1 = (chunk_ids[preceeding_pos], new_token) if preceeding_pos >= 0 else None
        following_pair = (merge_pair[1], chunk_ids[following_pos]) if following_pos < len(chunk_ids) else None
        new_pair_2 = (new_token, chunk_ids[following_pos]) if following_pos < len(chunk_ids) else None
        # print(f"Updating pairs: {merge_pair} -> {new_pair_1}, {new_pair_2} in doc {cur_doc_idx} at pos {pos_idx}")
        # print(f"Preceding pair: {preceeding_pair}, Following pair: {following_pair}")

        if preceeding_pair is not None:
            all_freq_counts[preceeding_pair] -= 1
            if all_freq_counts[preceeding_pair] <= 0:
                del all_freq_counts[preceeding_pair]
            all_freq_counts[new_pair_1] += 1
            if (cur_doc_idx, preceeding_pos) in pair_locations[preceeding_pair]:
                pair_locations[preceeding_pair].remove((cur_doc_idx, preceeding_pos))
            pair_locations[new_pair_1].add((cur_doc_idx, preceeding_pos))
            
        if following_pair is not None:
            all_freq_counts[following_pair] -= 1
            if all_freq_counts[following_pair] <= 0:
                del all_freq_counts[following_pair]
            all_freq_counts[new_pair_2] += 1
            if (cur_doc_idx, following_pos) in pair_locations[following_pair]:
                pair_locations[following_pair].remove((cur_doc_idx, following_pos))
            pair_locations[new_pair_2].add((cur_doc_idx, following_pos))

        # Remove the merge pair from the pair locations
        all_freq_counts[merge_pair] -= 1
        if (cur_doc_idx, pos_idx) in pair_locations[merge_pair]:
            pair_locations[merge_pair].remove((cur_doc_idx, pos_idx))

        chunks[cur_doc_idx] = chunk_ids[:pos_idx] + [new_token] + chunk_ids[pos_idx + 2:]
    del pair_locations[merge_pair]
    del all_freq_counts[merge_pair]

    return pair_locations, chunks


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
        boundaries = find_chunk_boundaries(file, cpu_count, "<|endoftext|>".encode('utf-8'))

    all_doc_ids = []
    
    all_pair_locations = collections.defaultdict(set)
    with mp.Pool(processes=cpu_count) as pool:
        results = pool.starmap(
            get_chunk_ids,
            [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        for res in results:
            all_doc_ids.extend(res)


        res = pool.starmap(count_pairs, [(doc_id, chunk_ids) for doc_id, chunk_ids in enumerate(all_doc_ids)])
        for freq_count, pair_locations_res in res:
            for k, v in freq_count.items():
                all_freq_counts[k] += v
            for k, v in pair_locations_res.items():
                all_pair_locations[k].update(v)

            

    while len(vocab) < vocab_size:
        max_freq_pair = max(all_freq_counts.items(), key=lambda x: x[1])[0]
        idx = len(vocab)
        vocab[idx] = vocab[max_freq_pair[0]] + vocab[max_freq_pair[1]]
        merges.append((vocab[max_freq_pair[0]], vocab[max_freq_pair[1]]))

        all_pair_locations, all_doc_ids = update_counts(all_pair_locations, 
                                                        all_doc_ids, 
                                                        max_freq_pair, 
                                                        idx)
        
    return vocab, merges


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(f"Updated vocabulary: {vocab}")
    
