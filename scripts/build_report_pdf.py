"""Build the jury PDF from website/data/report.md + the results in website/data (headless Chrome).

    python scripts/build_report_pdf.py --out website/data/report.pdf
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website" / "data"
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", "google-chrome", "chromium"]

CSS = """
@page { size: A4; margin: 16mm 16mm 16mm 16mm; }
body { font: 10.2pt/1.45 "Segoe UI", Arial, sans-serif; color: #1a1d23; }
h1 { font-size: 20pt; margin: 0 0 2mm; } h2 { font-size: 12.5pt; margin: 5mm 0 1.5mm; color: #1f4e9c; break-after: avoid; page-break-after: avoid; }
table, .kpis { break-inside: avoid; page-break-inside: avoid; }
.sub { color: #555; margin-bottom: 4mm; } a { color: #1f4e9c; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 9.2pt; }
th, td { border: 1px solid #d5d9e0; padding: 1.2mm 2mm; text-align: left; } th { background: #eef2f8; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
.kpis { display: flex; gap: 3mm; margin: 3mm 0 2mm; }
.kpi { flex: 1; border: 1px solid #d5d9e0; border-radius: 2mm; padding: 2mm 3mm; }
.kpi b { display: block; font-size: 15pt; } .kpi span { color: #555; font-size: 8.5pt; }
ul, ol { margin: 1mm 0 2mm 5mm; padding-left: 3mm; } li { margin: 0.6mm 0; }
code { font-size: 9pt; background: #f2f4f7; padding: 0 1mm; }
"""


def rows_html(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(
        f'<td class="n">{r.get(k)}</td>' if isinstance(r.get(k), (int, float)) else f"<td>{html.escape(str(r.get(k, '')))}</td>"
        for _, k in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "report.pdf"))
    args = ap.parse_args()
    site = json.loads((DATA / "site.json").read_text(encoding="utf-8"))
    metrics = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
    pred = json.loads((ROOT / "predictions_samples.json").read_text(encoding="utf-8"))
    report = markdown.markdown((DATA / "report.md").read_text(encoding="utf-8"), extensions=["tables"])

    logs = pred.get("log", {})
    dur = sum(l["duration"] for l in logs.values())
    ratio = sum(l.get("part_a_sec", 0) + l.get("part_b_sec", 0) for l in logs.values()) / dur
    n_events = sum(len(v["events"]) for v in pred["videos"].values())
    alarms = 0
    for v in pred["videos"].values():
        prev = 0.0
        for _, r in v["risk"]:
            alarms += r >= 0.5 > prev
            prev = r
    dev = next((a for a in metrics["ablations"] if a["name"].startswith("Dev sets")), None)
    score = dev["rows"][0]["score_a"] if dev else metrics.get("score_a")

    team = site["team"]
    members = ", ".join(f"{m['name']} ({m['role']})" for m in team["members"])
    links = site["links"]
    per_class = [{"class": c, "F1@0.3": v["f1_03"], "F1@0.5": v["f1_05"], "F1@0.7": v["f1_07"],
                  "TP/FP/FN @0.5": f"{v['tp']}/{v['fp']}/{v['fn']}"} for c, v in metrics["per_class"].items()]
    timing = [{"video": l, "length s": round(v["duration"], 1), "Part A s": v.get("part_a_sec"),
               "Part B s": v.get("part_b_sec"), "limit s": v["budget_sec"],
               "x video": round((v.get("part_a_sec", 0) + v.get("part_b_sec", 0)) / v["duration"], 2)}
              for l, v in logs.items()]
    tables = ""
    for a in metrics["ablations"]:
        cols = [("Variant", "variant"), ("Score A", "score_a"), ("micro F1@0.5", "micro_f1_05")]
        if any("perception_x_realtime" in r for r in a["rows"]):
            cols.append(("perception x realtime", "perception_x_realtime"))
        tables += f"<h2>{html.escape(a['name'])}</h2>" + rows_html(a["rows"], cols)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>RoadSense — technical report</h1>
<div class="sub">Team {html.escape(team['name'])}: {html.escape(members)}<br>
WIUT Hackathon 2026, Computer Vision track · code <a href="{links['repo']}">{links['repo']}</a> ·
website and live demo <a href="https://shoxijaxonbek-feba.static.hf.space">shoxijaxonbek-feba.static.hf.space</a></div>
<div class="kpis">
 <div class="kpi"><b>{score:.2f}</b><span>Score A on our own labels of C3902</span></div>
 <div class="kpi"><b>{n_events}</b><span>events in {dur / 60:.1f} min of 4K video (4 samples)</span></div>
 <div class="kpi"><b>{ratio:.1f}×</b><span>video length for Part A + B (limit 3×)</span></div>
 <div class="kpi"><b>{alarms}</b><span>false accident alarms on the samples</span></div>
</div>
{report}
<h2>Per-class results on the dev labels (C3902)</h2>
{rows_html(per_class, [(k, k) for k in ("class", "F1@0.3", "F1@0.5", "F1@0.7", "TP/FP/FN @0.5")])}
{tables}
<h2>Runtime of the official harness (RTX 4050 laptop GPU)</h2>
{rows_html(timing, [(k, k) for k in ("video", "length s", "Part A s", "Part B s", "limit s", "x video")])}
</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(doc, encoding="utf-8")
        for chrome in CHROME:
            try:
                subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                                f"--print-to-pdf={Path(args.out).resolve()}", page.as_uri()],
                               check=True, capture_output=True, timeout=120)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
