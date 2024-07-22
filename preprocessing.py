from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset

def preprocess_products(products):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['features'])
    return tfidf_matrix

def preprocess_ratings(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'product_id', 'rating']], reader)
    return train_test_split(data, test_size=0.2)
