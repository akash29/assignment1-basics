import regex as re
import multiprocessing as mp
from pretokenization_example import *
import collections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


vocab = {}
reverse_vocab = {}
for i in range(256):
    vocab[i] = bytes([i])
    reverse_vocab[bytes([i])] = i


def process_chunk(input_path, special_tokens, start, end, vocab, reverse_vocab):
    pattern = "|".join(map(re.escape, special_tokens))
    frequency = collections.defaultdict(int)
    # print(reverse_vocab)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunks = f.read(end - start).decode("utf-8", errors="ignore") 
        # pre-tokenization
        chunks = re.split(pattern, chunks)
        freq_words = []
        for chunk in chunks:
            words = list(map(lambda x: x.group(0),re.finditer(PAT, chunk)))
            for w1, w2 in zip(words, words[1:]):
                w = w1+w2
                temp = []
                for c in w:
                    enc_c = c.encode("utf-8")
                    idx = reverse_vocab[enc_c]
                    temp.append(idx)
                key = tuple(temp)
                frequency[key] += 1

       

            if len(frequency) > 0:
                most_frequent = max(frequency.items(), key = lambda x: (x[1], x[0]))[0]
                val = ''.join([vocab[i].decode("utf-8") for i in most_frequent])
                freq_words.append(val)
    return freq_words
                

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

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

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args = (input_path, special_tokens, start, end, vocab, reverse_vocab)
        with mp.Pool(processes=chunk_count) as pool:
            for res in pool.starmap(process_chunk, [args]):
                for word in res:
                    enc_word = word.encode("utf-8")
                    if enc_word not in reverse_vocab:
                        idx = len(vocab)
                        vocab[idx] = enc_word
                        reverse_vocab[enc_word] = idx
        print(f"vocab size: {len(vocab)}")
        print(f"New vocab: {vocab}")
            


        
if __name__ == "__main__":
    train_bpe("data/tiny_test.txt", 1000, ["<|endoftext|>"])




