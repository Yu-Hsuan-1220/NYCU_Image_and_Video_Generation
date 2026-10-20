"""
Collect results from outputs/ into report/ (a copy of template/LaTeX):
  - images -> report/Figure/{sds,vsd,sdi,diff_loss}/ and report/Figure/ablation/<setting>/<value>/
    (downscaled to 256px to keep the repo small; they are ~0.65in wide in the PDF)
  - CLIP scores -> report/Table/1_clip.tex and report/Table/2_ablation.tex

Usage: python scripts/make_report_assets.py
"""
import json
import os
import re
import shutil
from glob import glob

from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "outputs")
REPORT = os.path.join(ROOT, "report")
SIZE = 256

PROMPTS = [
    "A red bus driving on a desert road", "a boat in a river", "A cabin surrounded by forests",
    "A church beside a lake", "A villa close to the pool", "A castle next to a river",
    "A burger on the table", "A dog sitting on grass", "a cat sitting on a table", "A car on the road",
]
ABL_PROMPTS = ["A burger on the table", "a cat sitting on a table", "A villa close to the pool"]
DIFF_PROMPTS = ["A dog sitting on grass", "A castle next to a river", "A car on the road"]

# report setting name -> (outputs dir pattern, values)
ABLATIONS = {
    "sds_guidance_scale": ("ablation/sds_guidance/gs_{}", ["7.5", "15", "25", "50", "100"]),
    "sdi_inversion_n_steps": ("ablation/sdi_inv_steps/n_{}", ["1", "3", "5", "10", "25"]),
    "sdi_update_interval": ("ablation/sdi_update_interval/i_{}", ["25", "75", "200", "500", "1000"]),
}


def key(prompt):
    return prompt.replace(" ", "_")


def copy_img(src_dir, dst_dir, prompt):
    os.makedirs(dst_dir, exist_ok=True)
    img = Image.open(os.path.join(src_dir, f"{key(prompt)}.png")).convert("RGB")
    img.resize((SIZE, SIZE), Image.LANCZOS).save(os.path.join(dst_dir, f"{key(prompt)}.png"), optimize=True)


def clip_scores(d):
    """eval.json -> ({prompt_key: score}, average)."""
    data = json.load(open(os.path.join(OUT, d, "eval.json")))
    per = {os.path.basename(k)[:-4]: v for k, v in data.items() if k.endswith(".png")}
    return per, data["score"]


def avg_time(d):
    """Average optimization wall time (s) per prompt, parsed from tqdm in the per-prompt logs."""
    times = []
    for f in glob(os.path.join(OUT, d, "logs", "*.log")):
        m = re.findall(r"(\d+)/\1 \[([\d:]+)<", open(f, errors="ignore").read())
        if m:
            parts = [int(x) for x in m[-1][1].split(":")]
            times.append(sum(v * 60 ** i for i, v in enumerate(reversed(parts))))
    return sum(times) / len(times) if times else float("nan")


def main():
    # Seed report/ from the LaTeX template, never overwriting files that already exist
    # (so edits to main.tex / Section/*.tex survive a re-run)
    def copy_if_missing(src, dst):
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    shutil.copytree(os.path.join(ROOT, "template", "LaTeX"), REPORT, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("main.pdf"), copy_function=copy_if_missing)

    # ---- images ----
    for m in ["sds", "vsd", "sdi"]:
        for p in PROMPTS:
            copy_img(os.path.join(OUT, m), os.path.join(REPORT, "Figure", m), p)
    for p in DIFF_PROMPTS:
        copy_img(os.path.join(OUT, "diff_loss"), os.path.join(REPORT, "Figure", "diff_loss"), p)
    for setting, (pattern, values) in ABLATIONS.items():
        for v in values:
            for p in ABL_PROMPTS:
                copy_img(os.path.join(OUT, pattern.format(v)),
                         os.path.join(REPORT, "Figure", "ablation", setting, v), p)

    # ---- Table 1: per-prompt CLIP ----
    scores = {m: clip_scores(m) for m in ["sds", "vsd", "sdi"]}
    rows = []
    for p in PROMPTS:
        vals = [scores[m][0][key(p)] for m in ["sds", "vsd", "sdi"]]
        best = max(vals)
        cells = [f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}" for v in vals]
        rows.append(f"    {p:<34s} & " + " & ".join(cells) + r" \\")
    avgs = [scores[m][1] for m in ["sds", "vsd", "sdi"]]
    avg_cells = [f"\\textbf{{{v:.4f}}}" if v == max(avgs) else f"{v:.4f}" for v in avgs]
    table1 = (
        "\\begin{table}[t]\n  \\centering\n  \\footnotesize\n"
        "  \\caption{\\textbf{CLIP score} ($\\uparrow$, ViT-B/32) per prompt. Best per row in bold.}\n"
        "  \\label{tab:clip}\n  % values: outputs/{sds,vsd,sdi}/eval.json\n"
        "  \\begin{tabular}{@{}lccc@{}}\n    \\toprule\n    Prompt & SDS & VSD & SDI \\\\\n    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\midrule\n    \\textbf{Average}                   & " + " & ".join(avg_cells)
        + " \\\\\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )
    open(os.path.join(REPORT, "Table", "1_clip.tex"), "w").write(table1)

    # ---- Table 2: ablation CLIP (+ SDI runtime) ----
    titles = {
        "sds_guidance_scale": "(a) SDS: \\texttt{guidance\\_scale}",
        "sdi_inversion_n_steps": "(b) SDI: \\texttt{inversion\\_n\\_steps}",
        "sdi_update_interval": "(c) SDI: \\texttt{sdi\\_update\\_interval}",
    }
    blocks = []
    for setting, (pattern, values) in ABLATIONS.items():
        clips = [clip_scores(pattern.format(v))[1] for v in values]
        best = max(clips)
        clip_cells = [f"\\textbf{{{c:.4f}}}" if c == best else f"{c:.4f}" for c in clips]
        block = (
            f"    \\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{titles[setting]}}}}} \\\\\n"
            f"    Value & " + " & ".join(values) + " \\\\\n"
            f"    CLIP  & " + " & ".join(clip_cells) + " \\\\\n"
        )
        if setting.startswith("sdi"):
            # n_10 / i_25 were copied from outputs/sdi, so take their timing from there
            times = [avg_time("sdi" if (setting, v) in {("sdi_inversion_n_steps", "10"), ("sdi_update_interval", "25")}
                              else pattern.format(v)) for v in values]
            block += "    Time (s) & " + " & ".join(f"{t:.1f}" for t in times) + " \\\\\n"
        blocks.append(block)
    table2 = (
        "\\begin{table}[t]\n  \\centering\n  \\footnotesize\n  \\setlength{\\tabcolsep}{4pt}\n"
        "  \\caption{\\textbf{Ablation study.} Average CLIP score ($\\uparrow$) over the 10 prompts. "
        "Time: average optimization time per prompt on one RTX 4090.}\n"
        "  \\label{tab:ablation}\n  \\begin{tabular}{@{}lccccc@{}}\n    \\toprule\n"
        + "    \\midrule\n".join(blocks)
        + "    \\bottomrule\n  \\end{tabular}\n\\end{table}\n"
    )
    open(os.path.join(REPORT, "Table", "2_ablation.tex"), "w").write(table2)
    print(table1)
    print(table2)


if __name__ == "__main__":
    main()
