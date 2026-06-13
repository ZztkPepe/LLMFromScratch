from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
import json
import os

import regex as re

from cs336_basics.bpe import GPT2_PRETOKEN_PATTERN


def gpt2_bytes_to_unicode() -> dict[int, str]:
    visible = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    bytes_values = visible[:]
    code_points = visible[:]
    n = 0
    for byte in range(256):
        if byte not in bytes_values:
            bytes_values.append(byte)
            code_points.append(256 + n)
            n += 1
    return dict(zip(bytes_values, [chr(code_point) for code_point in code_points]))


def _load_gpt2_vocab(vocab_filepath: str | os.PathLike) -> dict[int, bytes]:
    byte_decoder = {value: key for key, value in gpt2_bytes_to_unicode().items()}
    with open(vocab_filepath, encoding="utf-8") as f:
        raw_vocab = json.load(f)
    return {
        token_id: bytes([byte_decoder[char] for char in token_text])
        for token_text, token_id in raw_vocab.items()
    }


def _load_gpt2_merges(merges_filepath: str | os.PathLike) -> list[tuple[bytes, bytes]]:
    byte_decoder = {value: key for key, value in gpt2_bytes_to_unicode().items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split()
            merges.append(
                (
                    bytes([byte_decoder[char] for char in left]),
                    bytes([byte_decoder[char] for char in right]),
                )
            )
    return merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        existing = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing:
                self.vocab[len(self.vocab)] = token_bytes
                existing.add(token_bytes)

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_token_set = set(self.special_tokens)
        self._special_pattern = self._build_special_pattern()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        return cls(_load_gpt2_vocab(vocab_filepath), _load_gpt2_merges(merges_filepath), special_tokens)

    def _build_special_pattern(self) -> re.Pattern | None:
        if not self.special_tokens:
            return None
        escaped = [re.escape(token) for token in self.special_tokens]
        return re.compile("(" + "|".join(escaped) + ")")

    def _split_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        if self._special_pattern is None:
            yield text, False
            return

        for part in self._special_pattern.split(text):
            if not part:
                continue
            yield part, part in self._special_token_set

    @lru_cache(maxsize=200_000)
    def _encode_pretoken(self, token_bytes: bytes) -> tuple[int, ...]:
        parts = tuple(bytes([byte]) for byte in token_bytes)
        if len(parts) == 1:
            return (self.token_to_id[parts[0]],)

        while len(parts) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(parts, parts[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            parts = self._merge_parts(parts, best_pair)

        return tuple(self.token_to_id[part] for part in parts)

    @staticmethod
    def _merge_parts(parts: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        merged: list[bytes] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        return tuple(merged)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for part, is_special in self._split_special_tokens(text):
            if is_special:
                token_ids.append(self.token_to_id[part.encode("utf-8")])
                continue

            for match in re.finditer(GPT2_PRETOKEN_PATTERN, part):
                token_ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
