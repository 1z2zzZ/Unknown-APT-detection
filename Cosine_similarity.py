import numpy as np


def cosine_similarity(vector1, vector2):
    # Convert lists to numpy arrays if they are not already
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Compute the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Compute the norms of the two vectors (i.e., their lengths)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    return cosine_similarity


def ncosine(x, y):
    cosine_sim = cosine_similarity(x, y)
    ncosine = (1 + cosine_sim) / 2
    return ncosine


def f(ncosine, nmi, omega):
    result = (omega * ncosine + (1 - omega) * nmi) / 2
    return result
