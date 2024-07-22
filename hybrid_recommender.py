from collaborative_filtering import get_collaborative_recommendations
from content_based_filtering import get_content_recommendations

def hybrid_recommendations(user_id, product_id, algo, trainset, cosine_sim, products, n=10):
    collab_recs = get_collaborative_recommendations(user_id, algo, trainset, n)
    content_recs = get_content_recommendations(product_id, cosine_sim, products, n)
    combined_recs = collab_recs + [(product['product_id'], 0) for product in content_recs.to_dict('records')]
    seen = set()
    unique_recs = []
    for rec in combined_recs:
        if rec[0] not in seen:
            seen.add(rec[0])
            unique_recs.append(rec)
        if len(unique_recs) >= n:
            break
    return unique_recs

