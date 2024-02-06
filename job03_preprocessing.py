import pandas as pd
from konlpy.tag import Okt
import re # 정규 표현식 / 정규 표현식은 문자열의 패턴을 찾고 조작하는 강력한 도구



df = pd.read_csv('./reviews_hotel.csv')
df.info()

df_stopwords = pd.read_csv('../../../Users/PC1/Downloads/stopwords.csv')
stopwords = list(df_stopwords['stopword'])
# stopwords = stopwords + ['영화', '감독', '연출', '배우', '연기', '작품', '관객', '장면']
okt = Okt()
cleaned_sentences = []
for review in df.reviews:
   review = re.sub('[^가-힣]', ' ', review)
   tokened_review = okt.pos(review, stem=True) #pos를 쓰면 형태소와 그것들의 품사가 각각 튜플로 묶여 리스트로 저장됨
   print(tokened_review)
   df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])
   df_token = df_token[(df_token['class']=='Noun') | (df_token['class']=='Verb') | (df_token['class']=='Adjective')]
   # 꾸며주거나 불필요한 단어들을 제거하기 위해 명사, 형용사, 부사만 남기고 나머지 단어들을 전부 제거해 줌
   words = []
   for word in df_token.word:
      if 1 < len(word): #한 글자 길이인 단어들 제거
         if word not in stopwords: #stopwords에 포함된 단어들 제거
            words.append(word)
   cleaned_sentence = ' '.join(words)
   cleaned_sentences.append(cleaned_sentence)
df['reviews'] = cleaned_sentences
df.dropna(inplace=True)
df.to_csv('./cleaned_reviews.csv', index=False)

print(df.head())
df.info()

# df = pd.read_csv('cleaned_reviews.csv')
# df.dropna(inplace=True)
# df.info()
# df.to_csv('./cleaned_reviews.csv', index=False)