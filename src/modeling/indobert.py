import torch

# Optimasi untuk 2-core (GitHub Actions). 
# Membatasi thread mencegah CPU "berantem" (contention).
torch.set_num_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

# Model Checkpoint (Indonesian RoBERTa Sentiment)
MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

class SentimentEngine:
    def __init__(self):
        print(f"[AI] Loading IndoBERT Model: {MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.eval() # Mode evaluasi (Read-only)
            print("[AI] Model loaded successfully!")
        except Exception as e:
            print(f"[AI] Failed to load model: {e}")
            self.model = None

    def predict(self, text: str) -> float:
        """
        Versi satu per satu (untuk kompatibilitas lama).
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[float]:
        """
        TURBO MODE: Memproses banyak teks sekaligus (Grosiran).
        """
        if not self.model or not texts:
            return [0.0] * len(texts)

        # Tokenisasi masif dengan padding otomatis
        encoded_input = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        # Eksekusi AI dalam satu tarikan napas
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Hitung Probabilitas
        logits = output.logits.numpy()
        probs = softmax(logits, axis=1) # [Baris, 3 Kolom]
        
        # Mapping: 0=Positive, 1=Neutral, 2=Negative
        # Rumus: Skor = Prob_Pos - Prob_Neg
        results = probs[:, 0] - probs[:, 2]
        
        return [float(s) for s in results]

# Global Instance
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = SentimentEngine()
    return _engine

if __name__ == "__main__":
    # Test
    eng = get_engine()
    texts = [
        "Laba bersih Astra International melonjak drastis tahun ini.", # Pos
        "Harga saham ASII anjlok parah karena skandal.",             # Neg
        "Astra merilis laporan keuangan kuartal 3 hari ini.",        # Neu
        "Penjualan mobil sedang lesu, tapi motor stabil."            # Mixed
    ]
    
    for t in texts:
        s = eng.predict(t)
        print(f"Text: {t}\nScore: {s:.4f}\n")
