import re
with open("wharton_verdict.txt", "r", encoding="utf-8") as file:
    text = file.read()
# print("Length of text:", len(text))
result = re.split(r'([.,:!?;+_"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
# print("Length of result:", len(result))
all_words = sorted(set(result))
vocab_size = len(all_words)
# print("Vocabulary size:", vocab_size)
vocab_dict = {word: i for i, word in enumerate(all_words)}
# for i, item in enumerate(vocab_dict.items()):
#     print(item)