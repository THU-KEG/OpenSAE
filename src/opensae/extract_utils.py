import os
import jieba
from transformers import AutoTokenizer

def refine_tokens(sentence: str, tokenizer: AutoTokenizer) -> dict:
    encoded = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    
    decoded_tokens = [tokenizer.decode(encoded["input_ids"][:i+1]) for i in range(len(encoded["input_ids"]))]
    refined_tokens = [decoded_tokens[0]]
    for i in range(1, len(decoded_tokens)):
        common_prefix_len = len(os.path.commonprefix([decoded_tokens[i-1], decoded_tokens[i]]))
        refined_tokens.append(decoded_tokens[i][common_prefix_len:])
    tokens = refined_tokens
    return {"tokens": tokens}