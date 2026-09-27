"""Models used in the Bilbo learning-curve comparison.

The paper's BOW model is an L2-regularized linear SVM on TF-IDF features.
`BOW_EXP` adds class-indicative unigram features selected by chi-square.  The
published KG expansion needs the original CN-DBpedia/LINE embeddings, which
are not distributed with this repository; this implementation leaves a clean
hook for supplying an embedding based expander separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC


class BowClassifier:
    """Paper baseline: TF-IDF unigrams plus a linear, L2 SVM."""

    def __init__(self, c: float = 1.0):
        self.model = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.classifier = LinearSVC(C=c)

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "BowClassifier":
        matrix = self.model.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return self.classifier.predict(self.model.transform(texts))


class BowExpansionClassifier:
    """Sparse BOW with chi-square selected class-indicative features.

    This is a reproducible approximation of BOW_EXP's feature-selection stage.
    It preserves normal TF-IDF features and appends the highest scoring features
    (`k_per_class * #classes`) before fitting the same LinearSVC.
    """

    def __init__(self, c: float = 1.0, k_per_class: int = 2):
        self.base = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.k_per_class = k_per_class
        self.selector: SelectKBest | None = None
        self.classifier = LinearSVC(C=c)

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "BowExpansionClassifier":
        labels = np.asarray(list(labels))
        matrix = self.base.fit_transform(texts)
        k = min(matrix.shape[1], self.k_per_class * len(np.unique(labels)))
        self.selector = SelectKBest(chi2, k=max(1, k)).fit(matrix, labels)
        from scipy.sparse import hstack

        self.classifier.fit(hstack([matrix, self.selector.transform(matrix)]), labels)
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        if self.selector is None:
            raise RuntimeError("Call fit before predict")
        from scipy.sparse import hstack

        matrix = self.base.transform(texts)
        return self.classifier.predict(hstack([matrix, self.selector.transform(matrix)]))


@dataclass
class BertClassifier:
    """Fine-tuned BERT-base Chinese multiclass classifier (paper's BERT arm)."""

    model_name: str = "bert-base-chinese"
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    seed: int = 0

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "BertClassifier":
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        texts, labels = list(texts), list(labels)
        self.labels_ = sorted(set(labels))
        self.label_to_id_ = {label: i for i, label in enumerate(self.labels_)}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)

        class EncodedDataset(Dataset):
            def __len__(self): return len(labels)
            def __getitem__(self, index):
                item = {key: torch.tensor(value[index]) for key, value in encodings.items()}
                item["labels"] = torch.tensor(self_label_ids[index])
                return item

        self_label_ids = [self.label_to_id_[label] for label in labels]
        self.tokenizer_ = tokenizer
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels_), id2label=dict(enumerate(self.labels_))
        )
        args = TrainingArguments(
            output_dir=".bert_checkpoints", per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate, num_train_epochs=self.epochs, seed=self.seed,
            save_strategy="no", logging_strategy="no", report_to=[]
        )
        Trainer(model=self.model_, args=args, train_dataset=EncodedDataset()).train()
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        import torch

        self.model_.eval()
        encoded = self.tokenizer_(list(texts), truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            prediction = self.model_(**encoded).logits.argmax(dim=-1).cpu().numpy()
        return np.asarray([self.labels_[index] for index in prediction])


def build_model(name: str, seed: int = 0):
    choices = {"bow": BowClassifier, "bow_exp": BowExpansionClassifier, "bert": lambda: BertClassifier(seed=seed)}
    if name not in choices:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(choices)}")
    return choices[name]()
