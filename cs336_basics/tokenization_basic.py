import collections

sample_text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

special_tokens = ["<|endoftext|>"]

frequency = collections.Counter(sample_text.split())


vocab = {}

for i in range(256):
    vocab[(chr(i),)] = 1

for token in special_tokens:
    vocab[(token,)] = 1

for token, count in frequency.items():
    vocab[tuple(token)] = count


def merge_tokens(vocab):

    merge_dict = {}

    for token, count in vocab.items():
        if token in special_tokens:
            continue
        for w1,w2 in zip(token, token[1:]):
            w = ''.join([w1, w2])
            if w not in merge_dict:
                merge_dict[w] = count
            else:
                merge_dict[w] += count

    max_val = max(merge_dict.items(), key=lambda x: (x[1], x[0]))

    token_to_remove = set()
    token_to_merge = set()

    for token,count in vocab.items():
        for idx, (w1,w2) in enumerate(zip(token, token[1:])):
            w = ''.join([w1, w2])
            if w == max_val[0]:
                new_token = token[:idx] + (w,) + token[idx+2:]
                token_to_merge.add((new_token, count))
                token_to_remove.add((token, count))
                break

    for token, count in token_to_remove:
        if token in vocab:
            del vocab[token]
    
    for token, count in token_to_merge:
        if token in vocab:
            vocab[token] += count
        else:
            vocab[token] = count


    return vocab

for i in range(6):
    print(f"Iteration {i}")
    vocab = merge_tokens(vocab)
    print(f"Merge {i}-{vocab}")


