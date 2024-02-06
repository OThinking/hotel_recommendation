import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import QStringListModel

form_window = uic.loadUiType('./hotel_recommendation.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Tfidf_matrix = mmread('./models/Tfidf_hotel_review.mtx').tocsr()
        with open('./models/tfidf.pickle', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_hotel_review.model')
        self.df_reviews = pd.read_csv('./cleaned_reviews_final.csv')
        self.names = list(self.df_reviews['names'])
        print(self.names)
        self.regions = list(self.df_reviews['regions'])
        print(self.regions)
        self.na_re = []
        for i in range(len(self.names)):
            self.na_re.append([self.names[i], self.regions[i]])
        print(self.na_re)
        regions = ['부산', '충청',  '강원', '경기', '경상', '인천', '제주', '전라', '서울']
        regions.sort()

        for region in regions:
            self.cmb_region.addItem(region)
        self.flag = 0
        self.cmb_region.currentIndexChanged.connect(self.select_regions)
        self.cmb_region.currentIndexChanged.connect(self.select_hotel)

        # self.btn_recommendation.clicked.connect(self.btn_slot)

    def select_regions(self):
        hotel_name = []
        region = self.cmb_region.currentText()
        self.cmb_hotel.clear()
        for i in range(len(self.names)):
            if self.na_re[i][1] == region:
                hotel_name.append(self.na_re[i][0])
        hotel_name.sort()
        for name in hotel_name:
            self.cmb_hotel.addItem(name)
        self.flag = 1

    def select_hotel(self):
        if self.flag == 1:
            sim_word = self.embedding_model.wv.most_similar(self.cmb_hotel.currentText(), topn=10)
            self.lbl_hotel.setText(sim_word)

    def btn_slot(self):
        key_word = self.le_keyword.text()
        if key_word in self.titles:
            recommendation = self.recommendation_by_movie_title(key_word)
        else:
            recommendation = self.recommendation_by_keyword(key_word)
        if recommendation:
            self.lbl_recommendation.setText(recommendation)

    def recommendation_by_hotel_names(self, title):
        movie_idx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movie_idx], self.Tfidf_matrix)
        recommendation = self.getRecommendation(cosine_sim)
        recommendation = '\n'.join(list(recommendation))
        return recommendation


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())