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

def retrieval_process(model, processor, database_features, query_images, num_similar_images=5):
    # Placeholder for similar images features
    similar_images_features = []

    clip_database_image_paths = [...]  # List of image paths in the CLIP database

    for image_path in clip_database_image_paths:
        inputs = processor(images=image_path, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        database_features.append(outputs.last_hidden_state)

    for query_image in query_images:
        # Compute similarity between query image and images in the CLIP database
        similarities = []
        for clip_image_features in database_features:
            # Compute similarity using cosine similarity
            similarity = torch.nn.functional.cosine_similarity(query_image.unsqueeze(0), clip_image_features.unsqueeze(0))
            similarities.append(similarity)

        # Get indices of top similar images
        top_similar_indices = torch.topk(torch.tensor(similarities), num_similar_images).indices

        # Retrieve features of top similar images from the CLIP database
        similar_images = [database_features[idx] for idx in top_similar_indices]
        similar_images_features.append(similar_images)

    # Return features of similar images
    return similar_images_features