import numpy as np

import torch
from torch.utils.data import Dataset


class DistributedTokenizedDataset(Dataset):
    def __init__(self, path, tokenizer, seq_length=4096, current_rank = 0, world_size = 1):
        assert seq_length is not None
        assert seq_length <= 4096, "The dataset is preprocessed to truncate documents at seq_length = 4096"
        
        self.seq_length = seq_length
        self.mmap = np.memmap(path, dtype = np.int32, mode = "r")
        
        self.current_rank = current_rank
        self.world_size = world_size
        
        self.tokenizer = tokenizer


    def __len__(self):
        return int(self.mmap.shape[0] / self.world_size / self.seq_length)


    def __getitem__(self, index):
        actual_index = index * self.world_size + self.current_rank
        start = self.seq_length * actual_index
        end = self.seq_length * (actual_index + 1)
        
        assert end <= self.mmap.shape[0], "Exceed current dataset"
        
        input_ids = torch.IntTensor(self.mmap[start:end])
        eos_pos = torch.where(input_ids == self.tokenizer.eos_token_id)[0] + 1
        
        if eos_pos.shape[0] == 0:   # TODO: data preprocessing bug.
            cu_seqlens = torch.tensor([0, self.seq_length])
        elif eos_pos[-1] != self.seq_length:
            cu_seqlens = torch.cat((torch.tensor([0]), eos_pos, torch.tensor([self.seq_length])))
        else:
            cu_seqlens = torch.cat((torch.tensor([0]), eos_pos))
        
        return (input_ids, cu_seqlens)


    def num_tokens(self):
        return self.position_bound - self.init_ptr
