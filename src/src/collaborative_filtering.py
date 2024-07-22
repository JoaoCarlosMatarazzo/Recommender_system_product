from surprise import KNNBasic, accuracy

def train_collaborative_model(trainset):
    algo = KNNBasic()
    algo.fit(trainset)
    return algo

def evaluate_collaborative_model(algo, testset):
    predictions = algo.test(testset)
    return accuracy.rmse(predictions)

def get_collaborative_recommendations(user_id, algo, trainset, n=10):
    all_products = trainset.all_items()
    product_ids = [trainset.to_raw_iid(iid) for iid in all_products]
    user_ratings = trainset.ur[trainset.to_inner_uid(user_id)]
    user_rated_products = [trainset.to_raw_iid(iid) for (iid, _) in user_ratings]
    products_to_recommend = [iid for iid in product_ids if iid not in user_rated_products]
    predictions = [algo.predict(user_id, iid) for iid in products_to_recommend]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [(pred.iid, pred.est) for pred in predictions[:n]]
