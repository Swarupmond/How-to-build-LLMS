from Dataloader import GPTdatasetv1
import torch
vocabulary_size = 100277
output_dim = 256
context_length = 4
# torch.manual_seed(123)  # For reproducibility
embedding_matrix = torch.nn.Embedding(vocabulary_size, output_dim)
with open("wharton_verdict.txt", "r") as file:
    text = file.read()
dataloader = GPTdatasetv1.create_dataloader(text, batch_size=8, maxlength=context_length, stride=context_length, shuffle=False)
data_iter = iter(dataloader)
input_ids, target_ids = next(data_iter)
# print("Input IDs:", input_ids)
print("shape of input IDs: ", input_ids.shape)
token_embeddings = embedding_matrix(input_ids)
# print("Token Embeddings:", token_embeddings)
print("shape of token embeddings: ", token_embeddings.shape)

pos_embeddings = torch.nn.Embedding(context_length, output_dim)
position_ids = torch.arange(context_length)
position_embeddings = pos_embeddings(position_ids)
# print("Position Embeddings:", position_embeddings)
print("shape of position embeddings: ", position_embeddings.shape)
vector_embeddings = token_embeddings + position_embeddings
print("Vector Embeddings:", vector_embeddings)
print("shape of vector embeddings: ", vector_embeddings.shape)