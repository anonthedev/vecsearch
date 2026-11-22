import matplotlib.pyplot as plt

latest = {}
with open("data/results.csv") as f:
    for line in f:
        name, recall, qps = line.strip().split(",")
        latest[name] = (float(recall), float(qps))

series = {}
brute = latest.get("brute")
for name, (recall, qps) in latest.items():
    if name == "brute" or name == "ivf":
        continue
    family = name.rsplit("-", 1)[0]
    series.setdefault(family, []).append((name, recall, qps))

for points in series.values():
    points.sort(key=lambda p: int(p[0].rsplit("-", 1)[-1]))

# Points near recall 0.99 share almost the same coordinates, so each label gets its own direction.
offsets = {
    "ivf64-32": (-18, 12),
    "ivf64-16": (8, -14),
    "ivf64-8": (8, 8),
    "ivf2048-512": (0, -16),
    "ivf2048-256": (16, -18),
    "ivf2048-128": (8, -12),
    "ivf4096-1024": (14, 12),
    "ivf4096-512": (-34, -16),
    "ivf4096-256": (18, -10),
    "ivf4096-128": (8, -12),
    "ivf8192-1024": (0, 14),
    "ivf8192-512": (-30, 18),
    "ivf8192-256": (8, 8),
    "ivf8192-128": (8, -12),
    "ivf16384-2048": (-42, 14),
    "ivf16384-1024": (28, -14),
    "ivf16384-512": (8, 10),
    "ivf16384-256": (8, -12),
    "ivf16384-128": (8, 8),
}

fig, (top, full) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
colors = {}

for ax in (top, full):
    if brute:
        ax.scatter([brute[1]], [brute[0]], marker="*", s=160, color="black", zorder=5, label="brute")
    for family, points in series.items():
        xs = [p[2] for p in points]
        ys = [p[1] for p in points]
        line, = ax.plot(xs, ys, marker="o", label=family)
        colors[family] = line.get_color()
    ax.set_xscale("log")
    ax.set_xlabel("queries per second")
    ax.set_ylabel("recall@10")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

top.set_xlim(6, 250)
top.set_ylim(0.86, 1.005)
top.set_title("recall 0.86 and above")
for family, points in series.items():
    for name, recall, qps in points:
        dx, dy = offsets.get(name, (6, 8))
        if recall < 0.86 or not 6 <= qps <= 250:
            continue
        top.annotate(
            name.rsplit("-", 1)[-1],
            (qps, recall),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            color=colors[family],
        )

full.set_ylim(0.35, 1.02)
full.set_title("all probes")
top.legend(loc="lower left")
fig.tight_layout()
fig.savefig("python-script/recall_qps.png", dpi=150)
