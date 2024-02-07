import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1])) #sorting 하면 인덱스가 깨지기 때문에 enumerate로 인덱스를 같이 받아준다
    print(simScore) # 결과값: [(0, 0.273097540404087) ... (10, 0.9999999999999994) ... (1328, 0.22230193603033996)]
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    # sorted()만 쓰면 리스트 아이템의 각 요소 순서대로 정렬
    # key 인자에 함수를 넘겨주면 해당 함수의 반환값을 비교하여 순서대로 정렬 x[0]=> 인덱스 값을 기준으로 x[1]=> 유사도 값을 기준으로
    # reverse=True를 주면 내림차순으로 정렬
    simScore = simScore[:11]
    print(simScore)
    movieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.iloc[movieIdx, 0]
    return recmovieList[1:11]

df_reviews = pd.read_csv('cleaned_reviews_final.csv')
Tfidf_matrix = mmread('./models/Tfidf_hotel_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 호텔 index 이용 추천
ref_idx = 10 # 10번째 있는 호텔
print(df_reviews.iloc[ref_idx, 0])
cosine_sim = linear_kernel(Tfidf_matrix[ref_idx], Tfidf_matrix)
# 코사인 유사도 값을 찾는다 / -1 근처로가면 아예 상관이 없는 1 근처로가면 연관이 있는 것을 구분
# 즉 여기서는 10번째 있는 호텔과 유사한 리뷰를 가지면 1로 아니면 -1로 수렴한다
print(cosine_sim) # 결과값: [[0.27309754 0.36106776 0.29252104 ... 0.20843067 0.24119297 0.22230194]]
print(type(cosine_sim)) # 결과값: <class 'numpy.ndarray'>
print(cosine_sim[0]) # 결과값: [0.27309754 0.36106776 0.29252104 ... 0.20843067 0.24119297 0.22230194]
print(cosine_sim[-1]) # 결과값: [0.27309754 0.36106776 0.29252104 ... 0.20843067 0.24119297 0.22230194]
print(len(cosine_sim)) # 결과값: 1
recommendation = getRecommendation(cosine_sim)
print(recommendation)

#keyword 이용 추천
# embedded_model = Word2Vec.load('./models/word2vec_hotel_review.model')
# keyword = '바다'
# sim_world = embedded_model.wv.most_similar(keyword, topn=10)
# words = [keyword]
# for word, _ in sim_world:
#     words.append(word)
# sentence = []
# count = 10
# for word in words:
#     sentence = sentence + [word] * count
#     count -= 1
# sentence = ' '.join(sentence)
# print(sentence)
# sentence_vec = Tfidf.transform([sentence])
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
#
# print(recommendation)