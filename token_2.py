import tiktoken
with open("wharton_verdict.txt", "r") as file:
    text = file.read()
#gpt4 tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
Integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
# print("Length of text:", len(text))
# for i in Integers:
#     print(Integers) 
# print(len(Integers))
Context_size = 4
for i in range(len(Integers)-Context_size):
    x = Integers[i:i+Context_size]
    y = Integers[i+1:i+1+Context_size]
    print("x:", tokenizer.decode(x))
    print("y:", tokenizer.decode(y))