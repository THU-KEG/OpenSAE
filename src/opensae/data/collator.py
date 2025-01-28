import torch

def packing_collate_fn(batch, max_length = 4096):
    input_ids = [x[0] for x in batch]
    
    input_ids = torch.cat(input_ids)
    
    cu_seqlens = list()
    for i, x in enumerate(batch):
        cu_seqlens.append(x[1] + i * max_length)
    cu_seqlens = torch.cat(cu_seqlens)
    
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlens = torch.max(seq_lens)
    
    return {
        "input_ids": input_ids.unsqueeze(0),
        "cu_seqlens": cu_seqlens.to(torch.int32),
        "max_seqlens": max_seqlens.to(torch.int32)
    }