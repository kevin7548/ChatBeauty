"""B_B4_misses.json -> experiments/B/bucket_recall.svg (학습 구간 리뷰 수 구간별 소스 Recall@100). 의존성 없이 SVG를 직접 쓴다."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "experiments/results/B_B4_misses.json").read_text())["recall@100_by_train_reviews"]
buckets = ["0", "1-4", "5-19", "20+"]
series = [("S1", "Fine-tuned BGE-M3", "#2a78d6"), ("S2", "Zero-shot BGE-M3", "#eb6834"), ("S3", "BM25 (raw reviews)", "#1baf7a")]

SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
W, H, L, R, T, B = 760, 440, 70, 30, 80, 70
Y1 = 0.7
y = lambda v: T + (Y1 - v) / Y1 * (H - T - B)
group_w = (W - L - R) / len(buckets)
bar_w, gap = 34, 2
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system,Helvetica,Arial,sans-serif">',
     f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
     f'<text x="{L}" y="26" font-size="15" fill="{INK}">Recall@100 by product review count (valid, 62,669 queries)</text>']
lx = L
for _, name, color in series:
    o.append(f'<rect x="{lx}" y="42" width="10" height="10" rx="2" fill="{color}"/>')
    o.append(f'<text x="{lx + 15}" y="51" font-size="12" fill="{MUTED}">{name}</text>')
    lx += 15 + len(name) * 6.6 + 24
for v in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    o.append(f'<line x1="{L}" x2="{W-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
    o.append(f'<text x="{L-8}" y="{y(v)+4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{v:.1f}</text>')
for gi, b in enumerate(buckets):
    cx = L + group_w * (gi + 0.5)
    x0 = cx - (len(series) * bar_w + (len(series) - 1) * gap) / 2
    for si, (key, name, color) in enumerate(series):
        v = d[b][key]
        bx, by, bh = x0 + si * (bar_w + gap), y(v), y(0) - y(v)
        tip = f'<title>{name}, {b} reviews: Recall@100 {v:.3f}</title>'
        o.append(f'<g><rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="{color}"/>'
                 f'<rect x="{bx:.1f}" y="{by + min(4, bh):.1f}" width="{bar_w}" height="{max(bh - 4, 0):.1f}" fill="{color}"/>{tip}</g>')
        o.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" font-size="11" fill="{INK}" text-anchor="middle">{v:.2f}</text>')
    o.append(f'<text x="{cx:.1f}" y="{H-B+18}" font-size="12" fill="{INK}" text-anchor="middle">{b}</text>')
    o.append(f'<text x="{cx:.1f}" y="{H-B+34}" font-size="11" fill="{MUTED}" text-anchor="middle">{d[b]["share_of_queries"]*100:.0f}% of queries</text>')
o.append(f'<text x="{(L+W-R)/2}" y="{H-10}" font-size="12" fill="{MUTED}" text-anchor="middle">Reviews of the ground-truth product in the train period (80%)</text>')
o.append(f'<text transform="translate(18,{(T+H-B)/2}) rotate(-90)" font-size="12" fill="{MUTED}" text-anchor="middle">Recall@100</text>')
o.append("</svg>")
out = ROOT / "experiments/B/bucket_recall.svg"
out.write_text("\n".join(o))
print(f"Saved {out}")
