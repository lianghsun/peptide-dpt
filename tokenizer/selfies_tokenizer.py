"""SELFIES tokenizer for Gemma-3-1b pretraining.

Each SELFIES atomic symbol (e.g. [C], [=N], [Ring1]) maps to one token ID.
Special tokens: PAD=0, BOS=1, EOS=2, UNK=3.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import selfies as sf


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class SelfiesTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.token2id: dict[str, int] = vocab
        self.id2token: dict[int, str] = {v: k for k, v in vocab.items()}

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_corpus(cls, selfies_list: List[str]) -> "SelfiesTokenizer":
        """Build vocab from a corpus of SELFIES strings."""
        symbols: set[str] = set()
        for s in selfies_list:
            try:
                symbols.update(sf.split_selfies(s))
            except Exception:
                continue
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for sym in sorted(symbols):
            if sym not in vocab:
                vocab[sym] = len(vocab)
        return cls(vocab)

    @classmethod
    def from_alphabet(cls) -> "SelfiesTokenizer":
        """Build vocab from SELFIES semantic-robust alphabet (no corpus needed)."""
        alphabet = sorted(sf.get_semantic_robust_alphabet())
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for sym in alphabet:
            if sym not in vocab:
                vocab[sym] = len(vocab)
        return cls(vocab)

    @classmethod
    def load(cls, path: str | Path) -> "SelfiesTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.token2id, f, indent=2)

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    def encode(
        self,
        selfies_str: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        try:
            symbols = list(sf.split_selfies(selfies_str))
        except Exception:
            return [BOS_ID, EOS_ID] if add_special_tokens else []

        ids = [self.token2id.get(s, UNK_ID) for s in symbols]

        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            tokens.append(self.id2token.get(i, ""))
        return "".join(tokens)

    def batch_encode(
        self,
        selfies_list: List[str],
        max_length: int = 256,
        padding: bool = True,
    ) -> dict:
        encoded = [
            self.encode(s, add_special_tokens=True, max_length=max_length)
            for s in selfies_list
        ]
        if padding:
            max_len = max(len(e) for e in encoded)
            attention_mask = [
                [1] * len(e) + [0] * (max_len - len(e)) for e in encoded
            ]
            encoded = [e + [PAD_ID] * (max_len - len(e)) for e in encoded]
        else:
            attention_mask = [[1] * len(e) for e in encoded]

        return {"input_ids": encoded, "attention_mask": attention_mask}


def build_and_save_vocab(output_path: str = "tokenizer/selfies_vocab.json") -> SelfiesTokenizer:
    """Build vocab from SELFIES alphabet and save. Run once before training."""
    tok = SelfiesTokenizer.from_alphabet()
    tok.save(output_path)
    print(f"Vocab size: {tok.vocab_size} → saved to {output_path}")
    return tok


if __name__ == "__main__":
    tok = build_and_save_vocab()
    # Smoke test
    test = "[C][C][=O][N][C@@H][Branch1][C][C][C][=O][O]"
    ids = tok.encode(test)
    recovered = tok.decode(ids)
    print(f"Original : {test}")
    print(f"Encoded  : {ids[:10]}...")
    print(f"Decoded  : {recovered}")
