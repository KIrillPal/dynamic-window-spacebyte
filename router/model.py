from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from transformers import AutoModel

from .config import RouterTrainConfig


class RouterModelProtocol(nn.Module):
    encoder: nn.Module
    classifier: nn.Module
    regressor: nn.Module

    def enable_gradient_checkpointing(self) -> None: ...

    def freeze_encoder(self, *, unfreeze_last_n_layers: int | None = None) -> None: ...

    def freeze_classifier_head(self) -> None: ...

    def freeze_regressor_head(self) -> None: ...


class AdaptiveComputeRouter(nn.Module):
    """MiniLM encoder with separate classification and regression heads."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        regression_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.regression_activation = regression_activation

        # Binary classification via 2-class logits + softmax (cross-entropy in trainer).
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 2),
        )
        # Last layer zero → logits [0,0] → softmax (0.5, 0.5) regardless of upstream features.
        _last = self.classifier[-1]
        assert isinstance(_last, nn.Linear)
        nn.init.zeros_(_last.weight)
        nn.init.zeros_(_last.bias)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, 1),
        )

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def freeze_encoder(self, *, unfreeze_last_n_layers: int | None = None) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

        if unfreeze_last_n_layers is None or unfreeze_last_n_layers <= 0:
            return

        layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if layers is None:
            return

        for layer in layers[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def freeze_classifier_head(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = False

    def freeze_regressor_head(self) -> None:
        for param in self.regressor.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        global_logits = self.classifier(pooled)
        window_ratio = self.regressor(pooled).squeeze(-1)

        if self.regression_activation == "sigmoid":
            window_ratio = torch.sigmoid(window_ratio)
        elif self.regression_activation == "clamp":
            window_ratio = window_ratio.clamp(0.0, 1.0)
        elif self.regression_activation != "none":
            raise ValueError(f"Unknown regression_activation={self.regression_activation!r}")

        return {
            "pooled": pooled,
            "global_logits": global_logits,
            "window_ratio": window_ratio,
        }


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp_min(1).cpu()
        x = self.input_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _packed_out, (h_n, _c_n) = self.lstm(packed)
        # Last layer forward/backward states are stable sequence summaries under padding.
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)


class BiLSTMRouter(nn.Module):
    """Bidirectional LSTM router with classification and regression heads."""

    def __init__(
        self,
        *,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        regression_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        pooled_size = hidden_size * 2
        self.regression_activation = regression_activation
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_size, pooled_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(pooled_size, 2),
        )
        _last = self.classifier[-1]
        assert isinstance(_last, nn.Linear)
        nn.init.zeros_(_last.weight)
        nn.init.zeros_(_last.bias)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_size, pooled_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(pooled_size, 1),
        )

    def enable_gradient_checkpointing(self) -> None:
        return None

    def freeze_encoder(self, *, unfreeze_last_n_layers: int | None = None) -> None:
        del unfreeze_last_n_layers
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_classifier_head(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = False

    def freeze_regressor_head(self) -> None:
        for param in self.regressor.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del token_type_ids
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        global_logits = self.classifier(pooled)
        window_ratio = self.regressor(pooled).squeeze(-1)

        if self.regression_activation == "sigmoid":
            window_ratio = torch.sigmoid(window_ratio)
        elif self.regression_activation == "clamp":
            window_ratio = window_ratio.clamp(0.0, 1.0)
        elif self.regression_activation != "none":
            raise ValueError(f"Unknown regression_activation={self.regression_activation!r}")

        return {
            "pooled": pooled,
            "global_logits": global_logits,
            "window_ratio": window_ratio,
        }


def build_router_model(
    cfg: RouterTrainConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> RouterModelProtocol:
    model_type = cfg.model_type.lower()
    if model_type in {"minilm", "adaptive_compute_router", "transformer"}:
        if cfg.model_name is None:
            raise ValueError("model_name must be set when model_type is minilm/transformer.")
        return AdaptiveComputeRouter(
            cfg.model_name,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            regression_activation=cfg.regression_activation,
        )
    if model_type == "bilstm":
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        vocab_size = len(tokenizer)
        return BiLSTMRouter(
            vocab_size=vocab_size,
            padding_idx=int(pad_token_id),
            embedding_dim=cfg.lstm_embedding_dim,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.dropout,
            hidden_dropout=cfg.hidden_dropout,
            regression_activation=cfg.regression_activation,
        )
    raise ValueError(f"Unknown model_type={cfg.model_type!r}")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom
