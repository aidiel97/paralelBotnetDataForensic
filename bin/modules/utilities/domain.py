import numpy as np

def cosine_similarity(array1, array2):
  dot_product = np.dot(array1, array2)
  norm_array1 = np.linalg.norm(array1)
  norm_array2 = np.linalg.norm(array2)
  similarity = dot_product / (norm_array1 * norm_array2)
  return similarity

# Sample arrays of integers
# arrays = [
#   np.array([1, 2, 3, 4, 5]),
#   np.array([2, 4, 6, 8, 10]),
#   np.array([3, 6, 9, 12, 15])
# ]
def meanOfSimilarity(arrays):
  if(len(arrays) > 1):
    # Calculate cosine similarity for each pair of arrays
    similarities = []
    for i in range(len(arrays)-1):
      for j in range(i+1, len(arrays)):
        similarity = cosine_similarity(arrays[i], arrays[j])
        similarities.append(similarity)

    # Calculate mean similarity value
    mean_similarity = np.mean(similarities)
    return mean_similarity
  else:
    return 1