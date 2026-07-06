from tokenclass import simpletokenizerv1
tokenizer = simpletokenizerv1(vocab)
text = "Hello, world!, this is Swarup Mondal.-- How are you all ?"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded) 