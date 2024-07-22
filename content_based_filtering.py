from sklearn.metrics.pairwise import linear_kernel

def get_content_recommendations(product_id, cosine_sim, products, n=10):
    idx = products[products['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    product_indices = [i[0] for i in sim_scores]
    return products.iloc[product_indices]
