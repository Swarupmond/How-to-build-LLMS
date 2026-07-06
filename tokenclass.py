import re
class simpletokenizerv1:
    def __init__(self, vocab):
        self.vocab = vocab
        #mapping for encoding
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        #reverse mapping for decoding
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}

    def encode(self, text):
        preprocessed = re.split(r'([.,:!?;+_"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [token if token in self.token_to_id else '<|UNK|>' for token in preprocessed]
        token_ids = [self.token_to_id[token] for token in preprocessed]
        return token_ids

    def decode(self, ids):
        text = ' '.join([self.id_to_token[id] for id in ids])
        
# join underscores with adjacent tokens
        text = re.sub(r'\s*_\s*', '_', text)
    # remove spaces after opening quote/bracket
        text = re.sub(r'([(\[{"])\s+', r'\1', text)
    # remove spaces before punctuation / closing quote
        text = re.sub(r'\s+([.,:;!?)\]}"])', r'\1', text)
    # normalize colon spacing for readability
        text = re.sub(r'\s*:\s*', ': ', text)
        return text

    