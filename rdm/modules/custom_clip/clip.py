import torch
from typing import List, Union
from rdm.modules.custom_clip.simple_tokenizer import \
    SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            print(f"WARNING: Input of length {len(tokens)} is too long for context length {context_length}. "
                  f"Cutting.")
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result