from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict

import os
import regex as re

from .pretokenization_example import find_chunk_boundaries

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_PRETOKENIZATION_WORKERS = 4

def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    for base_value in range(256):
        vocab[len(vocab)] = bytes([base_value])
    return vocab


def _special_token_pattern(special_tokens: list[str]) -> str | None:
    # 把 special_tokens 转换成一个可用于正则匹配的 pattern
    # 这里必须是从长到短去匹配，否则短的会抢占长的的匹配
    if not special_tokens:
        return None
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return "|".join(escaped)


def _pretoken_counts_for_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    # 返回该 Chunk 内所有 pre-token 的频次统计表
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    
    counts: Counter[tuple[bytes, ...]] = Counter()
    special_pattern = _special_token_pattern(special_tokens) # 用 special tokens 组成re pattern
    chunks = re.split(special_pattern, text) if special_pattern else [text] # 用 re pattern 去切分 text
    
    for chunk in chunks:
        for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk): # 用 GPT-2 的方式去切分
            token_bytes = match.group().encode('utf-8')
            counts[tuple(bytes([byte]) for byte in token_bytes)] += 1
    return counts


def _pretoken_counts(
    input_path: str | os.PathLike, 
    special_tokens: list[str]
) -> Counter[tuple[bytes, ...]]:
    """
    这里输出的是：{变成 bytes 的 word: 这个 word 出现的次数}
    """
    num_workers = min(MAX_PRETOKENIZATION_WORKERS, os.cpu_count() or 1)
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b""
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_workers if split_token else 1,
            split_special_token=split_token
        )
    
    chunks = list(zip(boundaries[:-1], boundaries[1:]))
    if not chunks:
        return Counter()
    if len(chunks) == 1: # 如果输入文本太短，或者只分了1个chunk，则不需要并行
        start, end = chunks[0]
        return _pretoken_counts_for_chunk(input_path, start, end, special_tokens)

    counts: Counter[tuple[bytes, ...]] = Counter()
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor: # 多进程并行
        futures = [
            executor.submit(_pretoken_counts_for_chunk, input_path, start, end, special_tokens)
            for start, end in chunks
        ]
        for future in futures:
            counts.update(future.result()) # 根据 key 去累加 value
    return counts


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes, ...] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    完成 bpe tokenizer 的训练，返回 vocab 和 merge
    """ 
    vocab: tuple[dict[int, bytes]] = _init_vocab(special_tokens) # 初始化 vocab
    merges: list[tuple[bytes, bytes]] = [] # 初始化 merge
    if vocab_size <= len(vocab):
        return dict(list(vocab.items())[:vocab_size]), merges
    
    word_counts: Counter[tuple[bytes, ...]] = _pretoken_counts(input_path, special_tokens) # pre-tokenization
    words: dict[int, tuple[bytes, ...]] = dict(enumerate(word_counts.keys()))
    counts: dict[int, int] = dict(enumerate(word_counts.values()))
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    # 这个变量存在的意义是：方便后面找到合并了这个pair之后，修改被这个pair合并所影响的words
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set) # 包含该 pair 的所有 word_id
    
    # 变更 word_id 对应的 pair_counts, pair_to_word_ids
    def update(word_id, word):
        count = counts[word_id]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(word_id)
    
    # 遍历所有pre-token好的序列，初始化 pair_counts 和 pair_to_word_ids
    # 至此之后 word_counts 不会再被调用，word 和 counts 不会再被更新
    # 我们只需要维护 pair_counts 和 pair_to_word_ids
    for word_id, word in words.items():
        update(word_id, word)
    
    # 移除某个 word 对所有 pair_counts 和 pair_to_word_ids 的贡献
    def remove_word_pairs(word_id: int, word: tuple[bytes, ...]) -> None:
        count = counts[word_id]
        for pair in zip(word, word[1:]):
            pair_counts[pair] -= count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
                pair_to_word_ids.pop(pair, None)
            else:
                pair_to_word_ids[pair].discard(word_id)
                
    # 开始训练 bpe tokenizer    
    while len(vocab) < vocab_size and pair_counts:
        # 找到改合并的 pair
        best_pair, _ = max(pair_counts.items(), key=lambda x: (x[1], x[0])) # 先比较 freq，如果 freq 相同再比较字典大小
        affected_word_ids = list(pair_to_word_ids.get(best_pair, ()))
        if not affected_word_ids:
            break
        
        # 合并pair
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        
        # 把其他存在这个 pair 的 word 中的 pair 也合并
        for word_id in affected_word_ids:
            old_word = words[word_id]
            if best_pair not in set(zip(old_word, old_word[1:])):
                continue
            remove_word_pairs(word_id, old_word) # 先把 old word 的相关 remove 掉
            new_word = _merge_word(old_word, best_pair)
            words[word_id] = new_word
            update(word_id, new_word) # 把新的 word 进行更新
                
    return vocab, merges
