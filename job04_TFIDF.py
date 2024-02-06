import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv('cleaned_reviews_final.csv')
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf=True) # tf: 한 문장안에서 단어가 얼마나 등장하느냐  df: 문서에 단어가 얼마나 언급되었는가
Tfidf_matrix = Tfidf.fit_transform(df_reviews['reviews'])
print(Tfidf_matrix.shape) # 행렬로 만들어줌

with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./models/Tfidf_hotel_review.mtx', Tfidf_matrix)