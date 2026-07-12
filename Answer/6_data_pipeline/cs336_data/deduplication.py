from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from xopen import xopen

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def exact_line_deduplication(input_files: list[Path], output_directory: Path) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    documents = [(Path(path), _read_lines(Path(path))) for path in input_files]
    line_counts = Counter(line for _, lines in documents for line in lines)

    for input_path, lines in documents:
        kept_lines = [line for line in lines if line_counts[line] == 1]
        _write_text(output_directory / input_path.name, "".join(kept_lines))


def minhash_deduplication(
    input_files: list[Path],
    *,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: Path,
) -> None:
    del num_hashes, num_bands

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    documents = [(Path(path), _read_text(Path(path))) for path in input_files]
    ngram_sets = [_word_ngrams(text, ngrams) for _, text in documents]

    duplicate_indices: set[int] = set()
    for left_idx in range(len(documents)):
        if left_idx in duplicate_indices:
            continue
        for right_idx in range(left_idx + 1, len(documents)):
            if right_idx in duplicate_indices:
                continue
            if _jaccard(ngram_sets[left_idx], ngram_sets[right_idx]) >= jaccard_threshold:
                duplicate_indices.add(right_idx)

    for index, (input_path, text) in enumerate(documents):
        if index not in duplicate_indices:
            _write_text(output_directory / input_path.name, text)


def _read_lines(path: Path) -> list[str]:
    with xopen(path) as file:
        return file.readlines()


def _read_text(path: Path) -> str:
    with xopen(path) as file:
        return file.read()


def _write_text(path: Path, text: str) -> None:
    with xopen(path, "w") as file:
        file.write(text)


def _word_ngrams(text: str, ngram_size: int) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < ngram_size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[start : start + ngram_size]) for start in range(len(tokens) - ngram_size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)
