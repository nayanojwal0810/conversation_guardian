"""
Lightweight, CPU-friendly GRU sequence model for ConversationGuardian (Task 5).

Learns directly from the chronological sequence of observed dialogue turns using
a word-token embedding, a unidirectional Gated Recurrent Unit (GRU) with padding/masking,
and a binary classification head.

Safeguards:
  - Vocabulary and preprocessing are built strictly on training data.
  - Padded and masked appropriately via torch.nn.utils.rnn.pack_padded_sequence.
  - Early stopping based strictly on validation Average Precision (AP).
  - Fixed seeds for reproducible execution on CPU.
"""

import re
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens and significant punctuation marks.
    Preserves contractions and emotional punctuation (!, ?, etc.).
    """
    clean = str(text or "").lower()
    return re.findall(r"\b[\w'-]+\b|[!?,.]", clean)


class Vocabulary:
    """
    Manages token-to-index mapping fitted strictly on training data.
    Special tokens:
      <pad> = 0: padding index (masked in embedding)
      <unk> = 1: unknown / out-of-vocabulary token
      <turn> = 2: explicit turn boundary token
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    TURN_TOKEN = "<turn>"

    PAD_IDX = 0
    UNK_IDX = 1
    TURN_IDX = 2

    def __init__(self, max_size: int = 5000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.TURN_TOKEN: self.TURN_IDX,
        }
        self.idx_to_token: Dict[int, str] = {v: k for k, v in self.token_to_idx.items()}
        self.is_fitted: bool = False

    def fit(self, tokenized_sequences: List[List[str]]) -> "Vocabulary":
        """Build vocabulary from tokenized training sequences only."""
        counter: Counter[str] = Counter()
        for seq in tokenized_sequences:
            counter.update(seq)

        # Retain top tokens meeting min_freq
        for token, count in counter.most_common(self.max_size - len(self.token_to_idx)):
            if count >= self.min_freq and token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token

        self.is_fitted = True
        return self

    def encode_tokens(self, tokens: List[str], max_len: int) -> Tuple[List[int], int]:
        """
        Encode a list of tokens into integer indices, with padding or truncation.
        Returns: (padded_indices, actual_length)
        """
        if not self.is_fitted:
            raise RuntimeError("Vocabulary must be fitted before encoding tokens.")

        encoded = [self.token_to_idx.get(t, self.UNK_IDX) for t in tokens]
        actual_len = len(encoded)

        if actual_len >= max_len:
            return encoded[:max_len], max_len
        elif actual_len == 0:
            # Handle empty sequence gracefully
            return [self.PAD_IDX] * max_len, 1
        else:
            padded = encoded + [self.PAD_IDX] * (max_len - actual_len)
            return padded, actual_len

    def __len__(self) -> int:
        return len(self.token_to_idx)


class SequenceTextDataset(Dataset):
    """PyTorch Dataset for multi-turn dialogue token sequences."""

    def __init__(
        self,
        token_sequences: List[List[str]],
        labels: Optional[np.ndarray],
        vocab: Vocabulary,
        max_len: int = 256,
    ):
        self.token_sequences = token_sequences
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

        self.encoded_ids: List[List[int]] = []
        self.lengths: List[int] = []

        for seq in token_sequences:
            ids, length = self.vocab.encode_tokens(seq, max_len=self.max_len)
            self.encoded_ids.append(ids)
            self.lengths.append(max(length, 1))

    def __len__(self) -> int:
        return len(self.encoded_ids)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.tensor(self.encoded_ids[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, length, y
        return x, length


class ConversationGRU(nn.Module):
    """
    Lightweight GRU sequence model architecture.
    Processes tokens chronologically with masking and early stopping.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Vocabulary.PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Unidirectional preserves natural temporal sequence
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with packed sequence for variable-length dialogues.
        Returns: logits (batch_size,)
        """
        emb = self.dropout(self.embedding(input_ids))
        # Clamp lengths to at least 1 and within input_ids length
        seq_len = input_ids.size(1)
        valid_lengths = torch.clamp(lengths.cpu(), min=1, max=seq_len)
        packed = pack_padded_sequence(
            emb,
            valid_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        h_last = h_n[-1]  # final layer hidden state
        logits = self.classifier(h_last).squeeze(-1)
        return logits


class RecurrentSequenceClassifier:
    """
    Scikit-learn compatible wrapper for training and evaluating ConversationGRU.
    Fits vocabulary strictly on train, trains on CPU with early stopping on validation AP,
    and returns continuous predicted probabilities.
    """

    def __init__(
        self,
        max_vocab_size: int = 5000,
        min_freq: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 15,
        patience: int = 3,
        max_length: int = 256,
        random_state: int = 42,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_length = max_length
        self.random_state = random_state

        self.vocab: Optional[Vocabulary] = None
        self.model: Optional[ConversationGRU] = None
        self.is_fitted: bool = False

        # Training metadata
        self.training_time_sec: float = 0.0
        self.epochs_trained: int = 0
        self.best_epoch: int = 0
        self.best_val_ap: float = 0.0
        self.trainable_params: int = 0

    @staticmethod
    def _messages_to_tokens(messages: List[str]) -> List[str]:
        """Convert an ordered list of message texts into a single sequence with turn markers."""
        tokens: List[str] = []
        for m in messages:
            tokens.append(Vocabulary.TURN_TOKEN)
            tokens.extend(tokenize_text(m))
        return tokens

    def get_trainable_params(self) -> int:
        """Return the total number of trainable model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(
        self,
        X_seqs: List[List[str]],
        y_train: np.ndarray,
        X_val_seqs: Optional[List[List[str]]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RecurrentSequenceClassifier":
        """
        Fit vocabulary on train sequences only, and train the GRU with early stopping on validation AP.
        """
        start_time = time.time()
        # Set reproducible seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 1. Tokenize train and fit vocabulary strictly on train
        train_tokens = [self._messages_to_tokens(msgs) for msgs in X_seqs]
        self.vocab = Vocabulary(max_size=self.max_vocab_size, min_freq=self.min_freq)
        self.vocab.fit(train_tokens)

        # 2. Build datasets and data loaders
        train_dataset = SequenceTextDataset(
            train_tokens, y_train, self.vocab, max_len=self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        has_val = (X_val_seqs is not None) and (y_val is not None)
        val_loader = None
        if has_val:
            val_tokens = [self._messages_to_tokens(msgs) for msgs in X_val_seqs]
            val_dataset = SequenceTextDataset(
                val_tokens, y_val, self.vocab, max_len=self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # 3. Instantiate model and optimizer
        self.model = ConversationGRU(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.trainable_params = self.get_trainable_params()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        best_weights = None
        best_metric = -1.0
        epochs_no_improve = 0

        # 4. Training loop
        for epoch in range(1, self.max_epochs + 1):
            self.epochs_trained = epoch
            self.model.train()
            train_loss_total = 0.0

            for batch_x, batch_len, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_len)
                loss = criterion(logits, batch_y)
                loss.backward()
                # Gradient clipping to stabilize recurrent updates
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss_total += loss.item() * len(batch_y)

            # Validation evaluation for early stopping
            if has_val:
                self.model.eval()
                val_preds: List[float] = []
                with torch.no_grad():
                    for v_x, v_len, _ in val_loader:
                        v_logits = self.model(v_x, v_len)
                        v_probs = torch.sigmoid(v_logits).cpu().numpy().tolist()
                        val_preds.extend(v_probs)

                val_ap = float(average_precision_score(y_val, val_preds))
                if val_ap > best_metric:
                    best_metric = val_ap
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    self.epochs_trained = epoch
                    break
            else:
                self.best_epoch = epoch
                self.epochs_trained = epoch

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            self.best_val_ap = best_metric
        else:
            self.epochs_trained = self.max_epochs

        self.training_time_sec = time.time() - start_time
        self.is_fitted = True
        return self

    def predict_proba(self, X_seqs: List[List[str]]) -> np.ndarray:
        """
        Compute positive-class predicted probabilities.
        Returns: numpy array of shape (N, 2)
        """
        if not self.is_fitted or self.model is None or self.vocab is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        self.model.eval()
        tokens = [self._messages_to_tokens(msgs) for msgs in X_seqs]
        dataset = SequenceTextDataset(tokens, None, self.vocab, max_len=self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_list: List[float] = []
        with torch.no_grad():
            for bx, blen in loader:
                logits = self.model(bx, blen)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.extend(probs.tolist())

        p_pos = np.array(probs_list, dtype=float)
        p_neg = 1.0 - p_pos
        return np.column_stack([p_neg, p_pos])

    def predict(self, X_seqs: List[List[str]], threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at specified decision threshold."""
        probs = self.predict_proba(X_seqs)[:, 1]
        return (probs >= threshold).astype(int)

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture and training hyperparameters."""
        return {
            "model_type": "ConversationGRU",
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "vocab_size": len(self.vocab) if self.vocab is not None else 0,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "max_length": self.max_length,
            "random_state": self.random_state,
            "trainable_params": self.trainable_params,
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "best_val_ap": round(self.best_val_ap, 4),
            "training_time_sec": round(self.training_time_sec, 2),
        }
