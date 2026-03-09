"""SELFIES tokenizer for Gemma-3-1b pretraining.

Each SELFIES atomic symbol (e.g. [C], [=N], [Ring1]) maps to one token ID.
Special tokens: PAD=0, BOS=1, EOS=2, UNK=3.

Inherits from PreTrainedTokenizerBase so it can be passed as
processing_class to TRL's GRPOTrainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import selfies as sf
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class SelfiesTokenizer(PreTrainedTokenizerBase):
    """SELFIES tokenizer compatible with HuggingFace / TRL interfaces."""

    vocab_files_names: dict = {}

    def __init__(self, vocab: dict[str, int], **kwargs):
        # Set our vocab before super().__init__ so property overrides work
        self._selfies_vocab: dict[str, int] = vocab
        self._selfies_id2token: dict[int, str] = {v: k for k, v in vocab.items()}

        super().__init__(
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            padding_side="right",
            truncation_side="right",
            model_input_names=["input_ids", "attention_mask"],
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # HuggingFace required overrides                                       #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return len(self._selfies_vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._selfies_vocab)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        path = Path(save_directory) / "selfies_vocab.json"
        with open(path, "w") as f:
            json.dump(self._selfies_vocab, f, indent=2)
        return (str(path),)

    # Override token ID properties to always return our constants
    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    # Keep token2id / id2token as aliases for backward compat
    @property
    def token2id(self) -> dict[str, int]:
        return self._selfies_vocab

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_alphabet(cls) -> "SelfiesTokenizer":
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
            json.dump(self._selfies_vocab, f, indent=2)

    # ------------------------------------------------------------------ #
    # Core encoding / decoding                                            #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        selfies_str: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> List[int]:
        try:
            symbols = list(sf.split_selfies(selfies_str))
        except Exception:
            return [BOS_ID, EOS_ID] if add_special_tokens else []

        ids = [self._selfies_vocab.get(s, UNK_ID) for s in symbols]

        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        special = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        tokens = []
        for i in token_ids:
            if skip_special_tokens and i in special:
                continue
            tokens.append(self._selfies_id2token.get(i, ""))
        return "".join(tokens)

    def batch_decode(
        self,
        sequences: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors=None,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]
        ml = max_length or 512
        encoded = [self.encode(t, add_special_tokens=add_special_tokens, max_length=ml) for t in text]

        if padding:
            max_len = max(len(e) for e in encoded)
            attention_mask = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded]
            encoded = [e + [PAD_ID] * (max_len - len(e)) for e in encoded]
        else:
            attention_mask = [[1] * len(e) for e in encoded]

        data = {"input_ids": encoded, "attention_mask": attention_mask}

        if return_tensors == "pt":
            import torch
            data = {k: torch.tensor(v) for k, v in data.items()}

        return BatchEncoding(data)

    def batch_encode(
        self,
        selfies_list: List[str],
        max_length: int = 256,
        padding: bool = True,
    ) -> dict:
        return self(selfies_list, max_length=max_length, padding=padding)


def build_and_save_vocab(output_path: str = "tokenizer/selfies_vocab.json") -> SelfiesTokenizer:
    """Build vocab from SELFIES alphabet and save. Run once before training."""
    tok = SelfiesTokenizer.from_alphabet()
    tok.save(output_path)
    print(f"Vocab size: {tok.vocab_size} → saved to {output_path}")
    return tok


if __name__ == "__main__":
    tok = build_and_save_vocab()
    test = "[C][C][=O][N][C@@H][Branch1][C][C][C][=O][O]"
    ids = tok.encode(test)
    recovered = tok.decode(ids)
    print(f"Original : {test}")
    print(f"Encoded  : {ids[:10]}...")
    print(f"Decoded  : {recovered}")
