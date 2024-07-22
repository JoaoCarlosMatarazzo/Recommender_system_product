import unittest
from src.preprocessing import preprocess_products, preprocess_ratings
from src.data_loader import load_products, load_ratings

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_products(self):
        products = load_products('data/products.csv')
        tfidf_matrix = preprocess_products(products)
        self.assertEqual(tfidf_matrix.shape[0], products.shape[0])
        
    def test_preprocess_ratings(self):
        ratings = load_ratings('data/ratings.csv')
        trainset, testset = preprocess_ratings(ratings)
        self.assertGreater(len(trainset), 0)
        self.assertGreater(len(testset), 0)

if __name__ == '__main__':
    unittest.main()

