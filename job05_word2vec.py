# 단어를 백터화 해주는 작업
import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('cleaned_reviews_final.csv')
df_review.info()

reviews = list(df_review['reviews'])
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)
# vector_size: 차원의 수를 줄인다(학습을 용이하게 하기 위해), window: kernel_size와 같은 역할(window 역시 같은 의미)
# min_count:최소출현빈도, 이 수를 넘겨야 학습에 포함시키겠다, workers:학습 시 cpu 어느정도 쓸지
embedding_model.save('./models/word2vec_hotel_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))