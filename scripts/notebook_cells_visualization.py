# ============================================================
# CELLS MẪU - copy từng block vào Kaggle notebook
# Thêm các cells này SAU cell training (subprocess.run train.py)
# ============================================================

# ── CELL 1: Cài matplotlib (nếu chưa có) ──────────────────
!pip install -q matplotlib

# ── CELL 2: Visualize ngay sau khi training xong ──────────
import json
from pathlib import Path

# Tự tìm metrics.json mới nhất
base = Path("outputs/internvl3_2b")
candidates = sorted(base.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    print("Chưa có metrics.json. Chạy training trước!")
else:
    METRICS_PATH = str(candidates[0])
    print(f"Đọc: {METRICS_PATH}")

    import subprocess
    subprocess.run([
        "python", "scripts/visualize_training.py",
        "--metrics", METRICS_PATH,
        "--save", "training_plot.png",
    ], check=True)

    # Hiển thị ảnh trong notebook
    from IPython.display import Image, display
    display(Image("training_plot.png"))

# ── CELL 3: Xem raw metrics (số liệu chi tiết) ────────────
with open(METRICS_PATH) as f:
    m = json.load(f)

print("=== EPOCH SUMMARY ===")
for ep in m["epoch_summary"]:
    print(f"  Epoch {ep['epoch']}: avg_train_loss = {ep['avg_train_loss']:.6f}")

print("\n=== VALIDATION LOSS ===")
if m["val_loss"]:
    for v in m["val_loss"]:
        print(f"  Epoch {v['epoch']} | Step {v['step']:>6}: {v['loss']:.6f}")
    best = min(m["val_loss"], key=lambda x: x["loss"])
    print(f"\n  Best val_loss = {best['loss']:.6f} @ step {best['step']}")
else:
    print("  (chưa có — cần chạy đến step 2000 mới có eval)")

print(f"\n=== TRAIN LOSS ===")
print(f"  Total steps logged: {len(m['train_loss'])}")
if m["train_loss"]:
    print(f"  First loss: {m['train_loss'][0]['loss']:.6f}")
    print(f"  Last  loss: {m['train_loss'][-1]['loss']:.6f}")

# ── CELL 4: Inline plot trong notebook (không cần subprocess) ──
import sys
sys.path.insert(0, ".")
from scripts.visualize_training import plot_metrics, print_summary

with open(METRICS_PATH) as f:
    metrics = json.load(f)

print_summary(metrics)
fig = plot_metrics(metrics, save_path="training_plot.png")
