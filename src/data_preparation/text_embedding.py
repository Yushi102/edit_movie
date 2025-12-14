"""
Text Embedding Module for Japanese Text Features

Converts Japanese text to numerical embeddings for model input.
"""
import numpy as np
import pandas as pd
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTextEmbedder:
    """
    Simple text embedder using character-level features
    
    For Japanese text, we use:
    1. Character count (文字数)
    2. Has hiragana (ひらがなあり)
    3. Has katakana (カタカナあり)
    4. Has kanji (漢字あり)
    5. Has punctuation (句読点あり)
    6. Text length normalized (正規化された長さ)
    
    This is a lightweight approach that doesn't require external models.
    """
    
    def __init__(self, max_length: int = 50):
        """
        Initialize text embedder
        
        Args:
            max_length: Maximum text length for normalization
        """
        self.max_length = max_length
        self.embedding_dim = 6
        
        logger.info(f"SimpleTextEmbedder initialized with {self.embedding_dim} dimensions")
    
    def _has_hiragana(self, text: str) -> bool:
        """Check if text contains hiragana"""
        return any('\u3040' <= c <= '\u309F' for c in text)
    
    def _has_katakana(self, text: str) -> bool:
        """Check if text contains katakana"""
        return any('\u30A0' <= c <= '\u30FF' for c in text)
    
    def _has_kanji(self, text: str) -> bool:
        """Check if text contains kanji"""
        return any('\u4E00' <= c <= '\u9FFF' for c in text)
    
    def _has_punctuation(self, text: str) -> bool:
        """Check if text contains punctuation"""
        punctuation = '、。！？!?,.，．'
        return any(c in punctuation for c in text)
    
    def embed_text(self, text: Optional[str]) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text (can be None or NaN)
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Handle missing text
        if text is None or pd.isna(text) or text == '':
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        text = str(text)
        
        # Extract features
        features = np.array([
            len(text),  # Character count
            float(self._has_hiragana(text)),  # Has hiragana
            float(self._has_katakana(text)),  # Has katakana
            float(self._has_kanji(text)),  # Has kanji
            float(self._has_punctuation(text)),  # Has punctuation
            min(len(text) / self.max_length, 1.0)  # Normalized length
        ], dtype=np.float32)
        
        return features
    
    def embed_series(self, text_series: pd.Series) -> np.ndarray:
        """
        Convert a pandas Series of text to embeddings
        
        Args:
            text_series: Series of text strings
        
        Returns:
            Array of shape (len(text_series), embedding_dim)
        """
        embeddings = []
        for text in text_series:
            embeddings.append(self.embed_text(text))
        
        return np.array(embeddings, dtype=np.float32)


class SentenceTransformerEmbedder:
    """
    Advanced text embedder using sentence-transformers
    
    This provides much richer semantic embeddings but requires
    the sentence-transformers library and a pre-trained model.
    """
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize sentence transformer embedder
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformerEmbedder initialized with {self.embedding_dim} dimensions")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def embed_text(self, text: Optional[str]) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text (can be None or NaN)
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Handle missing text
        if text is None or pd.isna(text) or text == '':
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        text = str(text)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_series(self, text_series: pd.Series) -> np.ndarray:
        """
        Convert a pandas Series of text to embeddings
        
        Args:
            text_series: Series of text strings
        
        Returns:
            Array of shape (len(text_series), embedding_dim)
        """
        # Replace NaN with empty string
        texts = text_series.fillna('').astype(str).tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)


def create_text_embedder(embedder_type: str = 'simple', **kwargs):
    """
    Factory function to create text embedder
    
    Args:
        embedder_type: Type of embedder ('simple' or 'transformer')
        **kwargs: Additional arguments for the embedder
    
    Returns:
        Text embedder instance
    """
    if embedder_type == 'simple':
        return SimpleTextEmbedder(**kwargs)
    elif embedder_type == 'transformer':
        return SentenceTransformerEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


if __name__ == "__main__":
    # Test simple embedder
    logger.info("Testing SimpleTextEmbedder...")
    simple_embedder = SimpleTextEmbedder()
    
    test_texts = [
        "こんにちは",
        "Hello World",
        "これは漢字です",
        "カタカナテスト",
        "混ざったText、です。",
        None,
        ""
    ]
    
    for text in test_texts:
        embedding = simple_embedder.embed_text(text)
        logger.info(f"Text: '{text}' -> Embedding shape: {embedding.shape}, values: {embedding}")
    
    # Test with pandas Series
    logger.info("\nTesting with pandas Series...")
    df = pd.DataFrame({'text': test_texts})
    embeddings = simple_embedder.embed_series(df['text'])
    logger.info(f"Series embeddings shape: {embeddings.shape}")
    logger.info(f"First embedding: {embeddings[0]}")
    
    logger.info("\n✅ Text embedding test complete!")
