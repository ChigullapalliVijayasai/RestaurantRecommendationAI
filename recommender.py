import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RestaurantRecommender:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.prepare_data()

    def prepare_data(self):
        self.data['combined_features'] = (
            self.data['Cuisine'].fillna('') + ' ' +
            self.data['Location'].fillna('') + ' ' +
            self.data['Restaurant_Name'].fillna('')
        )

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_matrix = self.vectorizer.fit_transform(self.data['combined_features'])

    def recommend(self, user_query, top_n=5):
        user_vec = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vec, self.feature_matrix)
        indices = similarities.argsort()[0][-top_n:][::-1]
        return self.data.iloc[indices][['Restaurant_Name', 'Location', 'Cuisine']].reset_index(drop=True)
