from fastapi import FastAPI
from pydantic import BaseModel
import random
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer

from function_timer import timeit

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


class InferenceRequest(BaseModel):
    input: str = ""
    
class InferenceResponse(BaseModel):
    # input: str = ""
    sentence_transformer_encode_time: float = 0.0
    sentence_transformer_embedding: list[float] = []


print("Loading Sentence Transformer...")
sentence_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@timeit
def encode_using_sentence_transformer(input: str):
    return sentence_embedding_model.encode(input)



@app.post("/api/infer/all")
def infer_all(request: InferenceRequest) -> InferenceResponse:
    print("Infering... ", request.input)
    sentence_transformer_embedding, sentence_transformer_encode_time = encode_using_sentence_transformer(request.input)
    
    
    return InferenceResponse(
        sentence_transformer_encode_time = sentence_transformer_encode_time,
        sentence_transformer_embedding = sentence_transformer_embedding
    )


# """
# server.py  —  Flask backend for the LSTM training web UI.

# Run:
#     pip install flask
#     python server.py

# Endpoints:
#     GET  /              → serve webui.html
#     POST /api/train     → start training (JSON config body)
#     GET  /api/stream    → SSE stream of training events
#     POST /api/stop      → request early stop
#     GET  /api/status    → current training state (polling fallback)
#     POST /api/infer/all → Perform Inference using all the models
# """

# import json
# import queue
# import threading
# import time
# import traceback
# from pathlib import Path

# import torch
# from flask import Flask, Response, jsonify, request, send_from_directory
# # from transformers import AutoTokenizer

# # from dataset_loader import KaggleSpamDataset, TokenizerTransform, ToTensor
# # from models import SimpleLSTM
# # from trainer import TorchTrainer
# # from torch.nn.utils.rnn import pad_sequence

# app = Flask(__name__)

# # ── Shared state ──────────────────────────────────────────────────────────────
# _state = {
#     "status": "idle",           # idle | running | stopped | done | error
#     "epoch": 0,
#     "total_epochs": 0,
#     "train_loss": [],
#     "val_loss": [],
#     "train_metrics": {},
#     "val_metrics": {},
#     "log": [],
#     "error": None,
# }
# _event_queue: queue.Queue = queue.Queue()
# _stop_flag = threading.Event()
# _lock = threading.Lock()


# def _push(event: str, data: dict):
#     with _lock:
#         _state["log"].append(f"[{event}] {json.dumps(data)}")
#     _event_queue.put({"event": event, "data": data})


# # def _pad_collate(batch, max_len=256):
# #     sequences, labels = zip(*batch)
# #     sequences = [s[:max_len] for s in sequences]
# #     padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# #     return padded, torch.stack(labels)


# # def _train_worker(cfg: dict):
# #     try:
# #         _push("status", {"status": "running", "msg": "Loading tokenizer…"})

# #         tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
# #         transform = torch.nn.Sequential(TokenizerTransform(tokenizer), ToTensor())

# #         _push("log", {"msg": f"Loading dataset from {cfg['data_path']}…"})
# #         dataset = KaggleSpamDataset(cfg["data_path"], transform=transform)

# #         n = len(dataset)
# #         train_n = int(n * cfg["train_split"])
# #         val_n = n - train_n
# #         train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])
# #         _push("log", {"msg": f"{n} samples → {train_n} train / {val_n} val"})

# #         collate = lambda b: _pad_collate(b, cfg["max_seq_len"])
# #         train_loader = torch.utils.data.DataLoader(
# #             train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate
# #         )
# #         val_loader = torch.utils.data.DataLoader(
# #             val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate
# #         )

# #         model = SimpleLSTM(
# #             input_size=cfg["vocab_size"],
# #             embedding_output_size=cfg["embedding_dim"],
# #             hidden_size=cfg["hidden_size"],
# #             num_layers=cfg["num_layers"],
# #             output_size=cfg["output_size"],
# #             dropout=cfg["dropout"],
# #         )
# #         params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #         _push("log", {"msg": f"Model ready — {params:,} parameters"})

# #         # Custom training loop that reports per-epoch progress
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #         model = model.to(device)
# #         criterion = torch.nn.CrossEntropyLoss()
# #         optimizer = torch.optim.Adam(
# #             model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
# #         )

# #         with _lock:
# #             _state["total_epochs"] = cfg["epochs"]
# #             _state["train_loss"] = []
# #             _state["val_loss"] = []

# #         for epoch in range(cfg["epochs"]):
# #             if _stop_flag.is_set():
# #                 _push("status", {"status": "stopped", "msg": "Training stopped by user."})
# #                 return

# #             model.train()
# #             epoch_loss = 0.0
# #             for X, y in train_loader:
# #                 X, y = X.to(device), y.to(device)
# #                 optimizer.zero_grad(set_to_none=True)
# #                 with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
# #                     loss = criterion(model(X), y)
# #                 loss.backward()
# #                 optimizer.step()
# #                 epoch_loss += loss.item()

# #             avg_train = epoch_loss / len(train_loader)

# #             model.eval()
# #             val_loss = 0.0
# #             correct = total = 0
# #             with torch.no_grad():
# #                 for X, y in val_loader:
# #                     X, y = X.to(device), y.to(device)
# #                     out = model(X)
# #                     val_loss += criterion(out, y).item()
# #                     preds = torch.argmax(out, dim=1)
# #                     correct += (preds == y).sum().item()
# #                     total += len(y)

# #             avg_val = val_loss / len(val_loader)
# #             val_acc = correct / total

# #             with _lock:
# #                 _state["epoch"] = epoch + 1
# #                 _state["train_loss"].append(round(avg_train, 4))
# #                 _state["val_loss"].append(round(avg_val, 4))

# #             _push("epoch", {
# #                 "epoch": epoch + 1,
# #                 "total": cfg["epochs"],
# #                 "train_loss": round(avg_train, 4),
# #                 "val_loss": round(avg_val, 4),
# #                 "val_acc": round(val_acc * 100, 2),
# #             })

# #         # Final evaluation
# #         _push("log", {"msg": "Running final evaluation…"})
# #         trainer = TorchTrainer(model)
# #         train_metrics = trainer.evaluate(train_loader)
# #         val_metrics = trainer.evaluate(val_loader)

# #         with _lock:
# #             _state["train_metrics"] = {k: round(float(v), 4) for k, v in train_metrics.items()}
# #             _state["val_metrics"] = {k: round(float(v), 4) for k, v in val_metrics.items()}

# #         torch.save(model.state_dict(), "outputs/lstm_model.pt")
# #         _push("status", {"status": "done", "msg": "Training complete. Model saved."})
# #         _push("metrics", {"train": _state["train_metrics"], "val": _state["val_metrics"]})

# #     except Exception as e:
# #         _push("status", {"status": "error", "msg": str(e)})
# #         _push("log", {"msg": traceback.format_exc()})


# # ── Routes ────────────────────────────────────────────────────────────────────

# @app.route("/")
# def index():
#     return send_from_directory("../webui/.svelte-kit/output/client/", "index.html")

# @app.route("/<path:path>")
# def home(path):
#     return send_from_directory('../webui/public', path)


# @app.route("/api/train", methods=["POST"])
# def start_training():
#     with _lock:
#         if _state["status"] == "running":
#             return jsonify({"error": "Training already in progress"}), 400
#         _state.update({"status": "running", "epoch": 0, "log": [],
#                         "train_metrics": {}, "val_metrics": {}})

#     _stop_flag.clear()
#     cfg = request.json
#     # thread = threading.Thread(target=_train_worker, args=(cfg,), daemon=True)
#     # thread.start()
#     return jsonify({"ok": True})


# @app.route("/api/stop", methods=["POST"])
# def stop_training():
#     _stop_flag.set()
#     return jsonify({"ok": True})


# @app.route("/api/status")
# def status():
#     with _lock:
#         return jsonify(dict(_state))


# @app.route("/api/stream")
# def stream():
#     def generate():
#         yield "retry: 1000\n\n"
#         while True:
#             try:
#                 item = _event_queue.get(timeout=15)
#                 yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
#             except queue.Empty:
#                 yield ": ping\n\n"

#     return Response(generate(), mimetype="text/event-stream",
#                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# @app.post("/api/infer/all")
# def infer_all():
#     data = request.json
#     print("Infering... ", data["input"])



# if __name__ == "__main__":
#     Path("outputs").mkdir(exist_ok=True)
#     print("Starting LSTM Training UI at http://localhost:5000")
#     app.run(debug=False, threaded=True, port=5000)

