import re
import ftfy
import html
import torch

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def retrieval_process(encoder, database_images, query_images, num_similar_images=5):
    database_features = []
    for image_batch in database_images:
        with torch.no_grad():
            features = encoder(image_batch)  # Assuming encoder is callable
        database_features.append(features)

    similar_images_features = []

    for query_image in query_images:
        query_features = []
        with torch.no_grad():
            features = encoder(query_image)  # Assuming encoder is callable
        query_features.append(features)

        # Compute similarity between query images and images in the database
        similar_images_feature = []
        for query_feature in query_features:
            similar_images = []
            for image_feature in database_features:
                # Compute similarity using cosine similarity
                similarities = torch.nn.functional.cosine_similarity(query_feature, image_feature, dim=1)
                # Get indices of top similar images
                top_similar_indices = torch.topk(similarities, num_similar_images).indices

                # Retrieve features of top similar images from the database
                similar_image = [image_feature[idx] for idx in top_similar_indices]
                similar_images.append(similar_image)
            similar_images_feature.append(similar_images)
        similar_images_features.append(similar_images_feature)

    # Return features of similar images
    return similar_images_features