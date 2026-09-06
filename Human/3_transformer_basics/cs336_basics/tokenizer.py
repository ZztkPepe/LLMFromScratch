

import regex as re
from typing import Iterator, Iterable

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""




class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab) # id to token
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        
        existing = set(self.vocab.values())
        # 防止 special_tokens 没有出现在传入的 vocab 中
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing:
                self.vocab[len(self.vocab)] = token_bytes
                existing.add(token_bytes)
        
        self.tok_to_id = {tok: tok_id for tok_id, tok in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair, in enumerate(self.merges)}
        self._special_token_set = set(self.special_tokens)
        self._special_pattern = self._special_token_pattern()
        
    def _special_token_pattern(self) -> re.Pattern | None:
        if not self.special_tokens:
            return None
        escaped = [re.escape(token) for token in self.special_tokens]
        return re.compile("(" + "|".join(escaped) + ")")

    def _split_by_special_tokens(self, text) -> Iterator[tuple[str, bool]]:
        if not self.special_tokens:
            yield text, False
            return
        
        for part in self._special_pattern.split(text):
            if not part:
                continue
            yield part, part in self._special_token_set
    
    @staticmethod
    def _merge_parts(parts: tuple[bytes, ...], pair: tuple[bytes, bytes]):
        i = 0
        merged: list[bytes] = []
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        return tuple(merged)
        
    def _encode_pretoken(self, token_bytes: bytes) -> tuple[int, ...]:
        # python 中遍历 bytes 时，迭代出来的是 int 而非 bytes。所以我们要把他变回 bytes
        parts: tuple[int, ...] = tuple(bytes([byte]) for byte in token_bytes)
        if len(parts) == 1:
            return (self.tok_to_id[parts[0]], )
        
        while len(parts) > 1: # 如果已经没有可 merge 的了就停
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(parts, parts[1:])
                if pair in self.merge_ranks
            ] # 先找到被 bpe merge 过的所有 pairs
            if not ranked_pairs: # 如果所有 bpe 认识的 bytes pair 都被 merge 过了 就返回
                break
            _, best_pair = min(ranked_pairs) # 找到 rank 值最小/优先级最高 的 pair
            parts = self._merge_parts(parts, best_pair) # 把优先级最高的 pair 从 parts 中 merge 了
            
        return tuple(self.tok_to_id[part] for part in parts)
        
    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for part, is_special in self._split_by_special_tokens(text):
            if is_special:
                token_ids.append(self.tok_to_id[part.encode('utf-8')])
                continue
            for match in re.finditer(GPT2_PRETOKEN_PATTERN, part):
                token_ids.extend(self._encode_pretoken(match.group().encode('utf-8')))
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[tok_id] for tok_id in ids) 
        return token_bytes.decode('utf-8', errors='replace')
        