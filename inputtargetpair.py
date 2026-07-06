import tiktoken
import torch
from Dataloader import GPTdatasetv1
with open("wharton_verdict.txt", "r") as file:
    text = file.read()
dataloder = GPTdatasetv1.create_dataloader(text, batch_size=8, maxlength=4, stride=4, shuffle=False)
data_iter = iter(dataloder)
input_ids, target_ids = next(data_iter)
print("Input IDs:", input_ids) 
print("Target IDs:", target_ids)
tokenizer = tiktoken.get_encoding("cl100k_base")
print("Decoded Input:", tokenizer.decode(input_ids[0].tolist()))
print("Decoded Target:", tokenizer.decode(target_ids[0].tolist()))
input_ids, target_ids = next(data_iter)
print("Input IDs:", input_ids) 
print("Target IDs:", target_ids)
tokenizer = tiktoken.get_encoding("cl100k_base")
print("Decoded Input:", tokenizer.decode(input_ids[0].tolist()))
print("Decoded Target:", tokenizer.decode(target_ids[0].tolist()))