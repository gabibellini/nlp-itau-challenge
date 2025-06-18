from sentence_transformers import SentenceTransformer

def load_model(model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
    """
    Load pre-trained model.
    """
    model = SentenceTransformer(model_name)
    return model

def generate_embeddings(model, texts: list):
    """
    generate embeddings
    """
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings
