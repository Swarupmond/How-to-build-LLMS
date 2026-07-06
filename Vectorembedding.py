import gensim.downloader as api
import numpy as np
model = api.load("Word2vec-google-news-300")
word_vector = model
print(word_vector['computer'])
print(word_vector(positive=['king', 'woman'], negative=['man'], topn=5))
#output possiblity is queen, princess, crown_prince, 
print(word_vector.similarity('computer', 'laptop'))

word1 = "man"
word2 = "woman"
word3 = "king"
word4 = "queen"
word5 = "computer"
word6 = "laptop"

vector_difference1 = model[word1] - model[word2]
vector_difference2 = model[word3] - model[word4]
vector_difference3 = model[word5] - model[word6]
magnitude1 = np.linalg.norm(vector_difference1)
magnitude2 = np.linalg.norm(vector_difference2)
magnitude3 = np.linalg.norm(vector_difference3)
print (f"Vector difference between '{word1}' and '{word2}': {vector_difference1}, Magnitude: {magnitude1}")
print (f"Vector difference between '{word3}' and '{word4}': {vector_difference2}, Magnitude: {magnitude2}")
print (f"Vector difference between '{word5}' and '{word6}': {vector_difference3}, Magnitude: {magnitude3}") 