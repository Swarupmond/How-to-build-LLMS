import re
from tokenclass import simpletokenizerv1
# text = "Hello, world!, this is Swarup Mondal.-- How are you all ?"
# result=re.split(r'([.,!?;+_"]|--|\s)', text)
# for item in result:
#     if item.strip():
#         print(item)
textfile = open("snowconnect_IMS_embedded_config.json", "r")
data = textfile.read()
preprocessed=re.split(r'([.,:!?;+_"()\']|--|\s)', data)
# result = [item for item in preprocessed if item.strip()]
# print(result) 
print("Total number of tokens:", len(preprocessed)) 
allwords = sorted(set(preprocessed))
vocab_size = len(allwords)
print("Vocabulary size:", vocab_size)
allwords.extend(['<|UNK|>', '<|endoftext|>'])
vocab = {word: index for index, word in enumerate(allwords)}
for i, item in enumerate(vocab.items()):
    print(item)

tokenizer = simpletokenizerv1(vocab)
text = '"table_label": "BMC IMAR",'
encoded = tokenizer.encode(text)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded) 
text1 = "Hello, swarup. are you an BMC employee ?"
encoded1 = tokenizer.encode(text1)
print("Encoded:", encoded1)
decoded1 = tokenizer.decode(encoded1)
print("Decoded:", decoded1) 
