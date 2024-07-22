import unittest
from src.data_loader import load_ratings, load_products

class TestDataLoader(unittest.TestCase):
    def test_load_ratings(self):
        ratings = load_ratings('data/ratings.csv')
        self.assertFalse(ratings.empty)
        
    def test_load_products(self):
        products = load_products('data/products.csv')
        self.assertFalse(products.empty)

if __name__ == '__main__':
    unittest.main()
