"""Step 5 - evaluate the trained model on the test split (RMSE, MAE, R2)."""
import json

from _common import embed, load, save

model = load("model.json")
test = load("test.json")
pred = [sum(a * b for a, b in zip(embed(g), model["weights"])) for g in test]
y = [g["y"] for g in test]
n = len(y)
rmse = (sum((p - t) ** 2 for p, t in zip(pred, y)) / n) ** 0.5
mae = sum(abs(p - t) for p, t in zip(pred, y)) / n
mean = sum(y) / n
ss_tot = sum((t - mean) ** 2 for t in y) or 1.0
r2 = 1 - sum((p - t) ** 2 for p, t in zip(pred, y)) / ss_tot
metrics = {"test_rmse": round(rmse, 4), "test_mae": round(mae, 4), "test_r2": round(r2, 4)}
save("metrics.json", metrics)
print(json.dumps(metrics))
print("MLTREE_METRIC " + json.dumps(metrics))  # picked up by mltree when the step is run from the map
