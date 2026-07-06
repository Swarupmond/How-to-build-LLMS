from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken
class GPTdatasetv1(Dataset):
    def __init__(self, data, tokenizer, maxlength, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(data, allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids) - maxlength, stride):
            x = token_ids[i:i + maxlength]
            y = token_ids[i + 1:i + maxlength + 1]
            self.input_ids.append(torch.tensor(x))
            self.target_ids.append(torch.tensor(y))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def create_dataloader(txt, batch_size=4, maxlength=256, stride=128, shuffle=True, drop_last=True, num_workers=0): 
        tokenizer = tiktoken.get_encoding("cl100k_base")
        dataset = GPTdatasetv1(txt, tokenizer, maxlength, stride)
        # dataloader takes the dataset, batch size - how many samples to load per batch, suffle - whether to shuffle the data at every epoch, drop_last - whether to drop the last incomplete batch, num_workers - how many subprocesses to use for data loading
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader
# with open("wharton_verdict.txt", "r") as file:
#     text = file.read()
# dataloder = GPTdatasetv1.create_dataloader(text, batch_size=8, maxlength=4, stride=1, shuffle=False)
# data_iter = iter(dataloder)
# input_ids, target_ids = next(data_iter)
# print("Input IDs:", input_ids) 
# print("Target IDs:", target_ids)
# tokenizer = tiktoken.get_encoding("cl100k_base")
# print("Decoded Input:", tokenizer.decode(input_ids[0].tolist()))
# print("Decoded Target:", tokenizer.decode(target_ids[0].tolist()))