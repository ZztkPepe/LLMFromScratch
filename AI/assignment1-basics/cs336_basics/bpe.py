from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import os

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for byte_value in range(256):
        vocab[len(vocab)] = bytes([byte_value])
    return vocab


def _special_token_pattern(special_tokens: list[str]) -> str | None:
    if not special_tokens:
        return None
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return "|".join(escaped)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    special_pattern = _special_token_pattern(special_tokens)
    chunks = re.split(special_pattern, text) if special_pattern is not None else [text]

    for chunk in chunks:
        for match in re.finditer(GPT2_PRETOKEN_PATTERN, chunk):
            token_bytes = match.group().encode("utf-8")
            counts[tuple(bytes([byte]) for byte in token_bytes)] += 1
    return counts


def _iter_pairs(word: tuple[bytes, ...]) -> Iterable[tuple[bytes, bytes]]:
    return zip(word, word[1:])


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
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
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab = _initial_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    if vocab_size <= len(vocab):
        return dict(list(vocab.items())[:vocab_size]), merges

    word_counts = _pretoken_counts(text, special_tokens)
    words = dict(enumerate(word_counts.keys()))
    counts = dict(enumerate(word_counts.values()))
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    def add_word_pairs(word_id: int, word: tuple[bytes, ...]) -> None:
        count = counts[word_id]
        for pair in _iter_pairs(word):
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(word_id)

    def remove_word_pairs(word_id: int, word: tuple[bytes, ...]) -> None:
        count = counts[word_id]
        for pair in _iter_pairs(word):
            pair_counts[pair] -= count
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
                pair_to_word_ids.pop(pair, None)
            else:
                pair_to_word_ids[pair].discard(word_id)

    for word_id, word in words.items():
        add_word_pairs(word_id, word)

    while len(vocab) < vocab_size and pair_counts:
        best_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        affected_word_ids = list(pair_to_word_ids.get(best_pair, ()))
        if not affected_word_ids:
            break

        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        for word_id in affected_word_ids:
            old_word = words[word_id]
            if best_pair not in set(_iter_pairs(old_word)):
                continue
            remove_word_pairs(word_id, old_word)
            new_word = _merge_word(old_word, best_pair)
            words[word_id] = new_word
            add_word_pairs(word_id, new_word)

    return vocab, merges
