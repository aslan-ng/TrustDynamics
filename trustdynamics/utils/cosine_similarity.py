import numpy as np


def cosine_similarity(vector_1, vector_2):
    result = None
    dot_product = np.dot(vector_1, vector_2)
    norm_vector_1 = np.linalg.norm(vector_1)
    norm_vector_2 = np.linalg.norm(vector_2)
    if norm_vector_1 == 0 or norm_vector_2 == 0:  # Avoid division by zero if any vector is zero
        result = 0
    else:
        result = float(dot_product / (norm_vector_1 * norm_vector_2))
    if result > 1:
        result = 1
    elif result < -1:
        result = -1
    return result


if __name__ == "__main__":
    vector_1 = [1, 2, 3]
    vector_2 = [3, 2, 1]

    similarity = cosine_similarity(vector_1, vector_2)
    print(f"Cosine similarity: {similarity}")