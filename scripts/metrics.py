import json
import os
import pickle
import numpy as np
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class VLMMetrics:
    def __init__(self, tfidf_path: str = "tfidf_vectorizer.pkl"):
        """
        Bộ đo lường TF-IDF và ROUGE chuyên dụng.
        """
        self.rouge_metric = evaluate.load("rouge")
        self.tfidf_path = tfidf_path
        self.vectorizer = None
        
        # Load vectorizer nếu có
        if os.path.exists(self.tfidf_path):
            print(f"[Info] Loading TF-IDF vectorizer from {self.tfidf_path}")
            with open(self.tfidf_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            print("[Warning] TF-IDF vectorizer not found. Will auto-fit on first use.")

    def _clean_text(self, text: str) -> str:
        """Làm sạch text, loại bỏ các thẻ XML nếu model sinh thừa"""
        text = text.strip()
        if "<answer>" in text:
            text = text.split("<answer>")[-1]
        if "</answer>" in text:
            text = text.split("</answer>")[0]
        return text.strip()

    def _extract_field(self, json_str: str, key: str = "instruction") -> str:
        """Parse JSON để lấy trường dữ liệu cụ thể (mặc định là 'instruction')"""
        try:
            clean_str = self._clean_text(json_str)
            data = json.loads(clean_str)
            return str(data.get(key, "")).strip()
        except json.JSONDecodeError:
            # Nếu model sinh lỗi format JSON, trả về chuỗi rỗng để phạt điểm
            return ""

    def fit_tfidf(self, corpus: List[str]):
        """Học từ vựng từ corpus"""
        print(f"Fitting TF-IDF on {len(corpus)} samples...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        self.vectorizer.fit(corpus)
        with open(self.tfidf_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ TF-IDF vectorizer saved to {self.tfidf_path}")

    def compute(self, predictions: List[str], references: List[str], target_field: str = "instruction") -> Dict[str, float]:
        """Tính toán điểm số"""
        # Trích xuất đúng trường target_field (instruction)
        pred_texts = [self._extract_field(p, key=target_field) for p in predictions]
        ref_texts = [self._extract_field(r, key=target_field) for r in references]
        
        if self.vectorizer is None:
            print("[Warning] TF-IDF not fitted. Auto-fitting on current reference data...")
            self.fit_tfidf(ref_texts)
        
        # 1. Tính TF-IDF Cosine Similarity
        try:
            tfidf_preds = self.vectorizer.transform(pred_texts)
            tfidf_refs = self.vectorizer.transform(ref_texts)
            
            similarities = []
            for i in range(len(pred_texts)):
                sim = cosine_similarity(tfidf_preds[i], tfidf_refs[i])[0, 0]
                similarities.append(sim)
            
            tfidf_score = np.mean(similarities) * 100
        except Exception as e:
            print(f"[Error] TF-IDF computation failed: {e}")
            tfidf_score = 0.0

        # 2. Tính ROUGE
        rouge_scores = self.rouge_metric.compute(
            predictions=pred_texts, 
            references=ref_texts, 
            use_stemmer=True
        )

        return {
            "ROUGE-1": rouge_scores['rouge1'] * 100,
            "ROUGE-2": rouge_scores['rouge2'] * 100,
            "ROUGE-L": rouge_scores['rougeL'] * 100,
            "TF-IDF": tfidf_score
        }