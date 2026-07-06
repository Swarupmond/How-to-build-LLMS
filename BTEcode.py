import tiktoken
# from transformers import AutoTokenizer 
text = "Hello, world!, this is Swarup Mondal.-- How are you all ? <|endoftext|> In the sunlit terraces of someunknownplace"
#gpt2 tokenizer
#tokenizer = tiktoken.get_encoding("gpt2")
#gpt4 tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base") 
Integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print("Encoded:", Integers)

string = tokenizer.decode(Integers)
print("Decoded:", string)
#llama3 tokenizer but need huggingface_hub
# tokenizer = AutoTokenizer.from_pretrained("llama3-7b-hf")
# inputs = tokenizer.encode(text, return_tensors="pt")
# print("Encoded with LLaMA3 tokenizer:", inputs) 
# string_decoded = tokenizer.decode(inputs[0])
# print("Decoded with LLaMA3 tokenizer:", string_decoded)