"""
server.py  —  Flask backend for the SpamLab training + inference web UI.

Run:
    pip install flask
    python server.py

Endpoints:
    GET  /                  → serve webui.html
    POST /api/train         → start training (JSON config body)
    GET  /api/stream        → SSE stream of training events
    POST /api/stop          → request early stop
    GET  /api/status        → current training state (polling fallback)
    POST /api/infer         → run inference on a text string (single model)
    POST /api/infer_all     → run ALL classifiers + return embeddings
    POST /api/embeddings    → return word2vec + sentence-transformer embeddings
    GET  /api/checkpoints   → list saved model files in outputs/
"""

import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, request, send_from_directory
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor
from models import SimpleLSTM, SimpleRNN, NeuralNetwork
from trainer import TorchTrainer

app = Flask(__name__)

# ── Shared training state ─────────────────────────────────────────────────────
_state = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": [],
    "val_loss": [],
    "train_metrics": {},
    "val_metrics": {},
    "log": [],
    "error": None,
    "last_trained": None,
}
_event_queue: queue.Queue = queue.Queue()
_stop_flag = threading.Event()
_lock = threading.Lock()

# ── Inference model cache ─────────────────────────────────────────────────────
_infer_cache: dict = {}
_infer_lock = threading.Lock()
_tokenizer_cache: dict = {}






from trainer import TorchTrainer, SklearnTrainer, Trainer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from typing import Callable, Any

sklearn_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    # ("Simple RNN", SimpleRNN, TorchTrainer, {"embedding_input_size": 30522}),
    ("SVM", svm.SVC, SklearnTrainer, {}),
    ("Nearest Neighbors (Euclidean)", KNeighborsClassifier, SklearnTrainer, { "metric": "euclidean" }),
    ("Nearest Neighbors (Minkowski)", KNeighborsClassifier, SklearnTrainer, { "metric": "minkowski" }),
    ("Nearest Neighbors (Cosine)", KNeighborsClassifier, SklearnTrainer, { "metric": "cosine" }),
    ("Naive Bayes", GaussianNB, SklearnTrainer, {}),
]

nn_classifiers = [
    ("Logistic Regression", LinearRegression, TorchTrainer, { "output_size": 6 }),
    ("Simple NN", NeuralNetwork, TorchTrainer, { "hidden_size": 512, "output_size": 6 }),
]

tokenizing_classifiers: list[tuple[str, Any, type[Trainer], dict[str, Any]]] = [
    # ("Simple RNN", SimpleRNN, TorchTrainer, { "output_size": 2 }),
    # ("Simple LSTM", SimpleLSTM, TorchTrainer, { "output_size": 2 }),
]















# ── Helpers ───────────────────────────────────────────────────────────────────

def _push(event: str, data: dict):
    with _lock:
        _state["log"].append(f"[{event}] {json.dumps(data)}")
    _event_queue.put({"event": event, "data": data})


def _pad_collate(batch, max_len=256):
    sequences, labels = zip(*batch)
    sequences = [s[:max_len] for s in sequences]
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.stack(labels)


def _get_tokenizer(name: str):
    if name not in _tokenizer_cache:
        _tokenizer_cache[name] = AutoTokenizer.from_pretrained(name)
    return _tokenizer_cache[name]


# ── Training worker ───────────────────────────────────────────────────────────

def _train_worker(cfg: dict):
    try:
        _push("status", {"status": "running", "msg": "Loading tokenizer…"})

        tokenizer = _get_tokenizer(cfg["tokenizer"])
        transform = torch.nn.Sequential(TokenizerTransform(tokenizer), ToTensor())

        _push("log", {"msg": f"Loading dataset from {cfg['data_path']}…"})
        dataset = KaggleSpamDataset(cfg["data_path"], transform=transform)

        n = len(dataset)
        train_n = int(n * cfg["train_split"])
        val_n = n - train_n
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])
        _push("log", {"msg": f"{n} samples → {train_n} train / {val_n} val"})

        collate = lambda b: _pad_collate(b, cfg["max_seq_len"])
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate
        )

        model = SimpleLSTM(
            input_size=cfg["vocab_size"],
            embedding_output_size=cfg["embedding_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            output_size=cfg["output_size"],
            dropout=cfg["dropout"],
        )
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _push("log", {"msg": f"Model ready — {params:,} parameters"})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _push("log", {"msg": f"Using device: {device}"})
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        with _lock:
            _state["total_epochs"] = cfg["epochs"]
            _state["train_loss"] = []
            _state["val_loss"] = []

        for epoch in range(cfg["epochs"]):
            if _stop_flag.is_set():
                _push("status", {"status": "stopped", "msg": "Training stopped by user."})
                return

            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    val_loss += criterion(out, y).item()
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)

            avg_val = val_loss / len(val_loader)
            val_acc = correct / total

            with _lock:
                _state["epoch"] = epoch + 1
                _state["train_loss"].append(round(avg_train, 4))
                _state["val_loss"].append(round(avg_val, 4))

            _push("epoch", {
                "epoch": epoch + 1,
                "total": cfg["epochs"],
                "train_loss": round(avg_train, 4),
                "val_loss": round(avg_val, 4),
                "val_acc": round(val_acc * 100, 2),
            })

        _push("log", {"msg": "Running final evaluation…"})
        trainer = TorchTrainer(model)
        train_metrics = trainer.evaluate(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        with _lock:
            _state["train_metrics"] = {k: round(float(v), 4) for k, v in train_metrics.items()}
            _state["val_metrics"] = {k: round(float(v), 4) for k, v in val_metrics.items()}
            _state["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        out_path = "outputs/lstm_model.pt"
        torch.save(model.state_dict(), out_path)

        with _infer_lock:
            _infer_cache.pop("lstm", None)

        _push("status", {"status": "done", "msg": f"Training complete. Model saved to {out_path}."})
        _push("metrics", {"train": _state["train_metrics"], "val": _state["val_metrics"]})

    except Exception as e:
        _push("status", {"status": "error", "msg": str(e)})
        _push("log", {"msg": traceback.format_exc()})


# ── Inference helpers ─────────────────────────────────────────────────────────

def _load_lstm(path: str = "outputs/lstm_model.pt", tokenizer_name: str = "bert-base-cased") -> SimpleLSTM:
    cache_key = f"lstm::{path}"
    with _infer_lock:
        if cache_key in _infer_cache:
            return _infer_cache[cache_key]
        tokenizer = _get_tokenizer(tokenizer_name)
        model = SimpleLSTM(input_size=tokenizer.vocab_size)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _infer_cache[cache_key] = model
        return model


def _load_rnn(path: str = "outputs/rnn_model.pt", tokenizer_name: str = "bert-base-cased") -> SimpleRNN:
    cache_key = f"rnn::{path}"
    with _infer_lock:
        if cache_key in _infer_cache:
            return _infer_cache[cache_key]
        tokenizer = _get_tokenizer(tokenizer_name)
        model = SimpleRNN(input_size=tokenizer.vocab_size)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _infer_cache[cache_key] = model
        return model


def _load_sentence_model():
    cache_key = "sentence_transformer"
    with _infer_lock:
        if cache_key in _infer_cache:
            return _infer_cache[cache_key]
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _infer_cache[cache_key] = m
        return m


def _load_word2vec_model():
    """Load Word2Vec model from outputs/word2vec.model if it exists."""
    cache_key = "word2vec"
    with _infer_lock:
        if cache_key in _infer_cache:
            return _infer_cache[cache_key]
        path = "outputs/word2vec.model"
        if not Path(path).exists():
            return None
        from gensim.models import Word2Vec
        m = Word2Vec.load(path)
        _infer_cache[cache_key] = m
        return m


def _load_sklearn_model(path: str):
    import pickle
    cache_key = f"sklearn::{path}"
    with _infer_lock:
        if cache_key in _infer_cache:
            return _infer_cache[cache_key]
        with open(path, "rb") as f:
            model = pickle.load(f)
        _infer_cache[cache_key] = model
        return model


def _tokenize_text(text: str, tokenizer_name: str = "bert-base-cased", max_len: int = 256):
    tokenizer = _get_tokenizer(tokenizer_name)
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    truncated = len(ids) > max_len
    ids = ids[:max_len]
    tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return tensor, len(ids), truncated


def _softmax_probs(logits: torch.Tensor) -> list:
    probs = F.softmax(logits, dim=-1)
    return [round(float(p), 4) for p in probs[0]]


def _infer_lstm(text: str) -> dict:
    model = _load_lstm()
    x, token_count, truncated = _tokenize_text(text)
    with torch.no_grad():
        logits = model(x)
    probs = _softmax_probs(logits)
    pred = int(torch.argmax(logits, dim=1).item())
    return {"prediction": pred, "confidences": probs, "token_count": token_count, "truncated": truncated}


def _infer_rnn(text: str) -> dict:
    model = _load_rnn()
    x, token_count, truncated = _tokenize_text(text)
    with torch.no_grad():
        logits = model(x)
    probs = _softmax_probs(logits)
    pred = int(torch.argmax(logits, dim=1).item())
    return {"prediction": pred, "confidences": probs, "token_count": token_count, "truncated": truncated}


def _infer_sklearn(text: str, embedding: np.ndarray, model_path: str) -> dict:
    clf = _load_sklearn_model(model_path)
    pred = int(clf.predict(embedding)[0])
    probs = [0.0, 0.0]
    if hasattr(clf, "predict_proba"):
        raw = clf.predict_proba(embedding)[0]
        probs = [round(float(p), 4) for p in raw]
    elif hasattr(clf, "decision_function"):
        score = float(clf.decision_function(embedding)[0])
        p_spam = float(1 / (1 + np.exp(-score)))
        probs = [round(1 - p_spam, 4), round(p_spam, 4)]
    else:
        probs[pred] = 1.0
    return {"prediction": pred, "confidences": probs}


def _get_word2vec_embedding(text: str) -> dict:
    """Return word2vec embedding: mean vector + per-token coverage."""
    import re
    tokens = [t for t in re.split(r"\W+", text.lower()) if t]

    model = _load_word2vec_model()
    if model is None:
        return {"available": False, "tokens": tokens, "vector": None, "coverage": 0, "vector_size": 128}

    vecs = []
    in_vocab = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
            in_vocab.append(True)
        else:
            in_vocab.append(False)

    if not vecs:
        mean_vec = [0.0] * model.wv.vector_size
    else:
        mean_vec = [round(float(v), 6) for v in np.mean(vecs, axis=0)]

    return {
        "available": True,
        "tokens": tokens,
        "in_vocab": in_vocab,
        "vector": mean_vec,
        "coverage": round(sum(in_vocab) / max(len(tokens), 1), 4),
        "vector_size": model.wv.vector_size,
    }


SKLEARN_PATHS = {
    "svm":     "outputs/svm_model.pkl",
    "knn_cos": "outputs/knn_cos_model.pkl",
    "knn_euc": "outputs/knn_euc_model.pkl",
    "nb":      "outputs/nb_model.pkl",
    "nn":      "outputs/nn_model.pt",
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../", "webui.html")


@app.route("/api/train", methods=["POST"])
def start_training():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Training already in progress"}), 400
        _state.update({
            "status": "running", "epoch": 0, "log": [],
            "train_metrics": {}, "val_metrics": {}
        })
    _stop_flag.clear()
    cfg = request.json
    thread = threading.Thread(target=_train_worker, args=(cfg,), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_training():
    _stop_flag.set()
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/stream")
def stream():
    def generate():
        yield "retry: 1000\n\n"
        while True:
            try:
                item = _event_queue.get(timeout=15)
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
            except queue.Empty:
                yield ": ping\n\n"

    return Response(
        generate(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.route("/api/infer", methods=["POST"])
def infer():
    body = request.json or {}
    text = (body.get("text") or "").strip()
    model_key = body.get("model", "lstm")
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        if model_key == "lstm":
            result = _infer_lstm(text)
        elif model_key == "rnn":
            result = _infer_rnn(text)
        elif model_key in ("svm", "knn_cos", "knn_euc", "nb"):
            path = SKLEARN_PATHS[model_key]
            if not Path(path).exists():
                return jsonify({"error": f"Model file not found: {path}"}), 404
            sm = _load_sentence_model()
            emb = sm.encode([text])
            result = _infer_sklearn(text, emb, path)
        elif model_key == "nn":
            path = SKLEARN_PATHS["nn"]
            if not Path(path).exists():
                return jsonify({"error": f"Model file not found: {path}"}), 404
            sm = _load_sentence_model()
            emb = sm.encode([text])
            emb_t = torch.tensor(emb, dtype=torch.float32)
            ck = f"nn::{path}"
            with _infer_lock:
                if ck not in _infer_cache:
                    nm = NeuralNetwork(input_size=emb.shape[1])
                    nm.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    nm.eval()
                    _infer_cache[ck] = nm
                nm = _infer_cache[ck]
            with torch.no_grad():
                logits = nm(emb_t)
            result = {"prediction": int(torch.argmax(logits, 1).item()), "confidences": _softmax_probs(logits)}
        else:
            return jsonify({"error": f"Unknown model: {model_key}"}), 400
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/infer_all", methods=["POST"])
def infer_all():
    """
    Run ALL classifiers + return embedding vectors.
    Body: { "text": str }
    Returns: { results: {model_key: {prediction, confidences, error?}}, embeddings: {...} }
    """
    body = request.json or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    results = {}
    sentence_embedding = None

    # Token-based models
    for key, loader_fn, path in [
        ("lstm", _infer_lstm, "outputs/lstm_model.pt"),
        ("rnn",  _infer_rnn,  "outputs/rnn_model.pt"),
    ]:
        if not Path(path).exists():
            results[key] = {"error": f"Not found: {path}"}
        else:
            try:
                results[key] = loader_fn(text)
            except Exception as e:
                results[key] = {"error": str(e)}

    # Sentence-embedding-based models — compute embedding once
    try:
        sentence_embedding = _load_sentence_model().encode([text])
    except Exception as e:
        sentence_embedding = None
        for key in ("svm", "knn_cos", "knn_euc", "nb", "nn"):
            results[key] = {"error": f"Could not load sentence model: {e}"}

    if sentence_embedding is not None:
        for key in ("svm", "knn_cos", "knn_euc", "nb"):
            path = SKLEARN_PATHS[key]
            if not Path(path).exists():
                results[key] = {"error": f"Not found: {path}"}
            else:
                try:
                    results[key] = _infer_sklearn(text, sentence_embedding, path)
                except Exception as e:
                    results[key] = {"error": str(e)}

        # Neural Net
        path = SKLEARN_PATHS["nn"]
        if not Path(path).exists():
            results["nn"] = {"error": f"Not found: {path}"}
        else:
            try:
                emb_t = torch.tensor(sentence_embedding, dtype=torch.float32)
                ck = f"nn::{path}"
                with _infer_lock:
                    if ck not in _infer_cache:
                        nm = NeuralNetwork(input_size=sentence_embedding.shape[1])
                        nm.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                        nm.eval()
                        _infer_cache[ck] = nm
                    nm = _infer_cache[ck]
                with torch.no_grad():
                    logits = nm(emb_t)
                results["nn"] = {
                    "prediction": int(torch.argmax(logits, 1).item()),
                    "confidences": _softmax_probs(logits),
                }
            except Exception as e:
                results["nn"] = {"error": str(e)}

    # Build embeddings payload
    embeddings = {}
    if sentence_embedding is not None:
        st_vec = [round(float(v), 6) for v in sentence_embedding[0]]
        embeddings["sentence_transformer"] = {
            "vector": st_vec,
            "dims": len(st_vec),
            "model": "all-MiniLM-L6-v2",
        }
    else:
        embeddings["sentence_transformer"] = {"error": "Model unavailable"}

    try:
        embeddings["word2vec"] = _get_word2vec_embedding(text)
    except Exception as e:
        embeddings["word2vec"] = {"error": str(e)}

    return jsonify({"results": results, "embeddings": embeddings})


@app.route("/api/embeddings", methods=["POST"])
def get_embeddings_route():
    body = request.json or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        vec = _load_sentence_model().encode([text])[0]
        st = {"vector": [round(float(v), 6) for v in vec], "dims": len(vec), "model": "all-MiniLM-L6-v2"}
    except Exception as e:
        st = {"error": str(e)}
    return jsonify({"sentence_transformer": st, "word2vec": _get_word2vec_embedding(text)})


@app.route("/api/checkpoints")
def checkpoints():
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    files = []
    for ext in ("*.pt", "*.pkl"):
        for f in sorted(outputs.glob(ext)):
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            files.append({"name": f.name, "path": str(f), "size_mb": size_mb})
    with _lock:
        last = _state.get("last_trained")
    return jsonify({"checkpoints": files, "last_trained": last})


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    print("=" * 56)
    print("  SpamLab — Training & Inference UI")
    print("  http://localhost:5000")
    print("=" * 56)
    app.run(debug=False, threaded=True, port=5000)