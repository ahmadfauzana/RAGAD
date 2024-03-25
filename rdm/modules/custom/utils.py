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

import torch

def retrieval_process(encoder, database_images, query_images, num_similar_images=5):
    print("Retrieval process started...")
    # Encode the database images using the encoder with the text prompt
    database_features = []
    for image in database_images:
        features = encoder(images=image.unsqueeze(0), text="image")['pixel_values']
        database_features.append(features)

    # Placeholder for similar images features
    similar_images_features = []

    # Encode the query images using the encoder with the text prompt
    query_features = []
    for query_image in query_images:
        features = encoder(images=query_image.unsqueeze(0), text="image")['pixel_values']
        query_features.append(features)

    # Compute similarity between query images and images in the database
    for query_feature in query_features:
        similarities = []
        for image_feature in database_features:
            # Compute similarity using cosine similarity
            similarity = torch.nn.functional.cosine_similarity(query_feature.unsqueeze(0), image_feature.unsqueeze(0))
            similarities.append(similarity.item())

        # Get indices of top similar images
        top_similar_indices = torch.topk(torch.tensor(similarities), num_similar_images).indices

        # Retrieve features of top similar images from the database
        similar_images = [database_features[idx] for idx in top_similar_indices]
        similar_images_features.append(similar_images)

    # Return features of similar images
    return similar_images_features


