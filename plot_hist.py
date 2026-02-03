import json, os, math
from collections import Counter
import matplotlib.pyplot as plt

path = "./logs/moe_routes.jsonl"

counts = Counter()
num_route_records = 0
top_k = None
layers = set()

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        if obj.get("type") == "meta":
            top_k = obj.get("top_k", top_k)
            continue
        if obj.get("type") != "route":
            continue

        layers.add(obj.get("layer"))
        ids = obj.get("topk_ids", [])
        for eid in ids:
            counts[int(eid)] += 1
        num_route_records += 1

if not counts:
    raise RuntimeError("No route records found in the JSONL file.")

# Build dense vectors 0..max_expert
max_expert = max(counts.keys())
x = list(range(max_expert + 1))
y = [counts.get(i, 0) for i in x]

# Total selections across all route records (this is tokens * top_k if every record has full k)
total_selections = sum(y)

# Normalized distribution p(e)
p = [(c / total_selections) if total_selections > 0 else 0.0 for c in y]

# Top-3 experts by count
top3 = counts.most_common(3)

# Entropy in bits: H = -sum p log2 p
entropy_bits = 0.0
for pi in p:
    if pi > 0:
        entropy_bits -= pi * math.log(pi, 2)

# Normalized entropy in [0,1] using support size (#experts with nonzero prob)
support = sum(1 for c in y if c > 0)
norm_entropy = (entropy_bits / math.log(support, 2)) if support > 1 else 0.0

# ----- Plot 1: raw histogram -----
plt.figure(figsize=(14, 5))
plt.bar(x, y)
plt.xlabel("Expert ID")
plt.ylabel("Selection count")

title = f"Expert selection histogram (route_records={num_route_records}, selections={total_selections}"
if top_k is not None:
    title += f", top_k={top_k}"
if len(layers) == 1:
    title += f", layer={next(iter(layers))}"
title += ")"
plt.title(title)

out_hist = "expert_hist.png"
plt.tight_layout()
plt.savefig(out_hist, dpi=200)
plt.close()

# ----- Plot 2: normalized distribution -----
plt.figure(figsize=(14, 5))
plt.bar(x, p)
plt.xlabel("Expert ID")
plt.ylabel("Selection probability p(expert)")
title2 = f"Normalized expert routing distribution (entropy={entropy_bits:.4f} bits, norm={norm_entropy:.4f})"
plt.title(title2)

out_norm = "expert_norm.png"
plt.tight_layout()
plt.savefig(out_norm, dpi=200)
plt.close()

# ----- Print summary -----
print(f"File: {path}")
print(f"Route records: {num_route_records}")
print(f"Total selections counted: {total_selections} (≈ tokens * top_k)")
print(f"Support (experts with nonzero count): {support}")
print(f"Entropy: {entropy_bits:.6f} bits")
print(f"Normalized entropy: {norm_entropy:.6f}  (0=peaked, 1=uniform over support)")

print("\nTop-3 experts:")
for eid, c in top3:
    pi = c / total_selections if total_selections else 0.0
    print(f"  expert {eid:>4}: count={c}, p={pi:.6%}")

print(f"\nSaved: {out_hist}")
print(f"Saved: {out_norm}")
