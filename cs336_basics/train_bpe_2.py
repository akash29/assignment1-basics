import regex as re
import multiprocessing as mp
from cs336_basics.pretokenization_example import *
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
    pair_locations = collections.defaultdict(set)
    for pos_idx, (u,v) in enumerate(zip(chunk_ids[:-1], chunk_ids[1:])):
        freq_map[(u, v)] += 1
        pair_locations[(u, v)].add((doc_id, pos_idx))
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


def update_counts(pair_locations, all_doc_ids, merge_pair, new_token, freq_heap):
    pair_location_list = pair_locations[merge_pair]

    pair_location_list = sorted(pair_location_list, reverse=True)

    for doc_idx, pos_idx in pair_location_list:
        doc = all_doc_ids[doc_idx]
        doc_copy = doc[:]
        if pos_idx < 0 or pos_idx+1>=len(doc):
            continue

        if pos_idx > 0:
            left_pair = (doc[pos_idx-1], doc[pos_idx])
            all_freq_counts[left_pair]-=1
            if all_freq_counts[left_pair] <=0:
                del all_freq_counts[left_pair]
            
            pair_locations[left_pair].discard((doc_idx, pos_idx-1))
            if not pair_locations[left_pair]:
                pair_locations.pop(left_pair, None)
        if pos_idx < len(doc)-2:
            right_pair = (doc[pos_idx+1], doc[pos_idx+2])
            all_freq_counts[right_pair]-=1
            if all_freq_counts[right_pair] <=0:
                del all_freq_counts[right_pair]
            
            pair_locations[right_pair].discard((doc_idx, pos_idx+1))
            if not pair_locations[right_pair]:
                pair_locations.pop(right_pair, None)
        
        doc_copy[pos_idx:pos_idx+2] = [new_token]
        updated_doc = doc_copy

        if pos_idx > 0:
            new_pair_1 = (updated_doc[pos_idx-1], new_token)
            all_freq_counts[new_pair_1]+=1
            count = all_freq_counts[new_pair_1]
            if len(freq_heap) > 0 and count > -freq_heap[0][0]:
                heapq.heappush(freq_heap, (-count, new_pair_1))
            else:
                heapq.heappush(freq_heap, (-count, new_pair_1))

            pair_locations[new_pair_1].add((doc_idx, pos_idx-1))
        if pos_idx < len(updated_doc)-1:
            new_pair_2 = (new_token, updated_doc[pos_idx+1])
            all_freq_counts[new_pair_2]+=1
            count = all_freq_counts[new_pair_2]
            if len(freq_heap) > 0 and count > -freq_heap[0][0]:
                heapq.heappush(freq_heap, (-count, new_pair_2))
            else:
                heapq.heappush(freq_heap, (-count, new_pair_2))
            pair_locations[new_pair_2].add((doc_idx, pos_idx))

        all_doc_ids[doc_idx] = doc_copy
    
    pair_locations.pop(merge_pair, None)
    all_freq_counts.pop(merge_pair, None)

    return pair_locations, all_doc_ids, freq_heap


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
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

    
    freq_count_heap = [(-v, k) for k,v in all_freq_counts.items()]
    heapq.heapify(freq_count_heap)

    while len(vocab) < vocab_size:
        max_freq_pair = None
        while freq_count_heap:
            count, pair = heapq.heappop(freq_count_heap)
            if all_freq_counts[pair] == -count:
                max_freq_pair = pair
                break
        
        if max_freq_pair is None:
            break

        idx = len(vocab)
        vocab[idx] = vocab[max_freq_pair[0]] + vocab[max_freq_pair[1]]
        merges.append((vocab[max_freq_pair[0]], vocab[max_freq_pair[1]]))

        all_pair_locations, all_doc_ids, freq_count_heap = update_counts(all_pair_locations, 
                                                        all_doc_ids, 
                                                        max_freq_pair, 
                                                        idx,
                                                        freq_count_heap)
        
        
        # if len(vocab) % 100 == 0:
        #     print(f"Merge {len(merges)}: {max_freq_pair}| Vocab size: {len(vocab)}")
        
       
        
    return vocab, merges


if __name__ == "__main__":
    input_path = "data/tiny_test.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(f"Updated vocabulary: {vocab}")
    
