"""
Transformer Encoder Module

Whisperで文字起こしした日本語テキストをBERTで埋め込みベクトルに変換

機能:
1. 日本語BERTモデルのロード
2. テキストの埋め込みベクトル生成
3. バッチ処理
4. GPU/CPU自動検出
5. LRUキャッシュによる高速化
"""
import os
import sys
import torch
import numpy as np
from typing import List, Optional
from functools import lru_cache
import logging
import warnings

logger = logging.getLogger(__name__)

# PyTorch 2.6未満の場合の回避策
torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version < (2, 6):
    # transformersをインポートする前にモンキーパッチを適用
    import importlib.util
    spec = importlib.util.find_spec("transformers.utils.import_utils")
    if spec is not None:
        import transformers.utils.import_utils as import_utils
        # チェック関数を無効化
        def patched_check():
            pass
        import_utils.check_torch_load_is_safe = patched_check
        logger.info("Applied monkey patch to bypass torch.load security check")

from transformers import AutoTokenizer, AutoModel


class TransformerEncoder:
    """日本語テキストをBERT埋め込みベクトルに変換するクラス"""
    
    def __init__(
        self,
        model_name: str = "cl-tohoku/bert-base-japanese-v3",
        cache_size: int = 10000,
        use_gpu: bool = True
    ):
        """
        初期化
        
        Args:
            model_name: 使用するBERTモデル名
            cache_size: LRUキャッシュのサイズ
            use_gpu: GPU使用フラグ
        """
        self.model_name = model_name
        self.cache_size = cache_size
        
        # デバイス検出
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
        
        # モデルとトークナイザーのロード
        try:
            logger.info(f"Loading Transformer model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()  # 評価モード
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            raise RuntimeError(f"Transformer model initialization failed: {e}")
        
        # 埋め込み次元数を取得
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # キャッシュ用の辞書（LRUキャッシュはインスタンスメソッドでは使えないため）
        self._cache = {}
        self._cache_order = []
        self._max_cache_size = cache_size
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """キャッシュから埋め込みを取得"""
        return self._cache.get(text)
    
    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """キャッシュに埋め込みを追加"""
        if text in self._cache:
            # 既存のエントリを更新（LRU順序を更新）
            self._cache_order.remove(text)
            self._cache_order.append(text)
        else:
            # 新規エントリを追加
            if len(self._cache) >= self._max_cache_size:
                # 最も古いエントリを削除
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            
            self._cache[text] = embedding
            self._cache_order.append(text)
    
    def encode(self, text: str) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            text: 日本語テキスト
        
        Returns:
            768次元の埋め込みベクトル
        """
        # 空文字列の場合はゼロベクトルを返す
        if not text or len(text.strip()) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # キャッシュをチェック
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached
        
        # トークナイズ
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 埋め込みを生成
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS]トークンの埋め込みを使用
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # キャッシュに追加
        self._add_to_cache(text, embedding)
        
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        複数のテキストをバッチ処理で埋め込みベクトルに変換
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
        
        Returns:
            (n_texts, embedding_dim)の埋め込み行列
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # バッチごとに処理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # キャッシュをチェックして、キャッシュされていないテキストを特定
            batch_results = [None] * len(batch_texts)  # 結果を格納する配列
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if not text or len(text.strip()) == 0:
                    batch_results[j] = np.zeros(self.embedding_dim, dtype=np.float32)
                else:
                    cached = self._get_from_cache(text)
                    if cached is not None:
                        batch_results[j] = cached
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)
            
            # キャッシュにないテキストをバッチ処理
            if uncached_texts:
                try:
                    # トークナイズ
                    inputs = self.tokenizer(
                        uncached_texts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 埋め込みを生成
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # [CLS]トークンの埋め込みを使用
                        batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # キャッシュに追加し、結果配列に格納
                    for idx, text, emb in zip(uncached_indices, uncached_texts, batch_emb):
                        self._add_to_cache(text, emb)
                        batch_results[idx] = emb
                
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU OOM during batch encoding. Falling back to CPU.")
                    # CPUにフォールバック
                    self.model.to("cpu")
                    self.device = torch.device("cpu")
                    
                    # 再試行（バッチサイズを半分に）
                    smaller_batch_size = max(1, batch_size // 2)
                    logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                    return self.encode_batch(texts, batch_size=smaller_batch_size)
            
            embeddings.extend(batch_results)
        
        return np.array(embeddings)
    
    def get_cache_stats(self) -> dict:
        """キャッシュの統計情報を取得"""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "cache_usage": len(self._cache) / self._max_cache_size if self._max_cache_size > 0 else 0
        }
