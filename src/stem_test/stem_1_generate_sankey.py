#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEM_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_STEM.jsonl"
DEFAULT_NON_STEM_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_non_STEM.jsonl"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "src" / "stem_test" / "stem_1_sankey_diagram.png"

METADATA_CLEAR_STEM_REASONS = {
    "nus_department_or_faculty_clear_stem",
    "ntu_department_clear_stem",
    "sutd_department_clear_stem",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object rows in {path}:{line_no}, got {type(row).__name__}")
            rows.append(row)
    return rows


def decision_path(row: dict) -> str:
    reason = str(row.get("scope_reason") or "")
    if reason in METADATA_CLEAR_STEM_REASONS:
        return "metadata_stem"
    if reason.startswith("excluded_"):
        return "excluded_non_stem"
    if reason == "paragraph_semantic_stem_override":
        return "paragraph_stem"
    if reason == "paragraph_semantic_non_stem_guard":
        return "paragraph_non_stem_guard"
    if reason == "stem_semantic_sentence_override":
        return "sentence_stem"
    if reason == "stem_keyword_override":
        return "keyword_stem"
    return "keyword_remainder"


def build_counts(rows: list[dict]) -> dict[str, int]:
    path_counts = Counter()

    for row in rows:
        path = decision_path(row)
        path_counts[path] += 1

    total = len(rows)
    metadata_stem = path_counts["metadata_stem"]
    excluded_non_stem = path_counts["excluded_non_stem"]
    semantic_entry = total - metadata_stem - excluded_non_stem

    paragraph_stem = path_counts["paragraph_stem"]
    paragraph_non_stem = path_counts["paragraph_non_stem_guard"]
    sentence_entry = semantic_entry - paragraph_stem - paragraph_non_stem

    sentence_stem = path_counts["sentence_stem"]
    keyword_entry = sentence_entry - sentence_stem

    keyword_stem = path_counts["keyword_stem"]
    keyword_remainder = path_counts["keyword_remainder"]
    final_stem = metadata_stem + paragraph_stem + sentence_stem + keyword_stem
    final_non_stem = total - final_stem

    return {
        "total": total,
        "metadata_stem": metadata_stem,
        "excluded_non_stem": excluded_non_stem,
        "semantic_entry": semantic_entry,
        "paragraph_stem": paragraph_stem,
        "paragraph_non_stem": paragraph_non_stem,
        "sentence_entry": sentence_entry,
        "sentence_stem": sentence_stem,
        "keyword_entry": keyword_entry,
        "keyword_stem": keyword_stem,
        "keyword_remainder": keyword_remainder,
        "final_stem": final_stem,
        "final_non_stem": final_non_stem,
    }


def _build_flow_links(counts: dict[str, int]) -> list[tuple[str, str, int]]:
    links = [
        ("All Modules", "Metadata clear STEM", counts["metadata_stem"]),
        ("All Modules", "Excluded non-STEM pattern", counts["excluded_non_stem"]),
        ("All Modules", "Enter semantic pipeline", counts["semantic_entry"]),
        ("Enter semantic pipeline", "Paragraph STEM override", counts["paragraph_stem"]),
        ("Enter semantic pipeline", "Paragraph non-STEM guard", counts["paragraph_non_stem"]),
        ("Enter semantic pipeline", "Enter sentence stage", counts["sentence_entry"]),
        ("Enter sentence stage", "Sentence STEM override", counts["sentence_stem"]),
        ("Enter sentence stage", "Enter keyword fallback", counts["keyword_entry"]),
        ("Enter keyword fallback", "Keyword STEM override", counts["keyword_stem"]),
        ("Enter keyword fallback", "Remain non-STEM or mixed", counts["keyword_remainder"]),
        ("Metadata clear STEM", "Final STEM", counts["metadata_stem"]),
        ("Paragraph STEM override", "Final STEM", counts["paragraph_stem"]),
        ("Sentence STEM override", "Final STEM", counts["sentence_stem"]),
        ("Keyword STEM override", "Final STEM", counts["keyword_stem"]),
        ("Excluded non-STEM pattern", "Final non-STEM", counts["excluded_non_stem"]),
        ("Paragraph non-STEM guard", "Final non-STEM", counts["paragraph_non_stem"]),
        ("Remain non-STEM or mixed", "Final non-STEM", counts["keyword_remainder"]),
    ]
    return [item for item in links if item[2] > 0]


def render_image_with_matplotlib(counts: dict[str, int], output_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path
    import numpy as np

    columns = [
        ["All Modules"],
        ["Metadata clear STEM", "Excluded non-STEM pattern", "Enter semantic pipeline"],
        ["Paragraph STEM override", "Paragraph non-STEM guard", "Enter sentence stage"],
        ["Sentence STEM override", "Enter keyword fallback"],
        ["Keyword STEM override", "Remain non-STEM or mixed"],
        ["Final STEM", "Final non-STEM"],
    ]

    links = _build_flow_links(counts)
    inflow = Counter()
    outflow = Counter()
    for src, tgt, val in links:
        outflow[src] += val
        inflow[tgt] += val
    node_value = {name: max(inflow[name], outflow[name]) for col in columns for name in col}
    total = counts["total"]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    node_w = 0.024
    col_x = np.linspace(0.04, 0.94 - node_w, len(columns))
    gap = 0.02
    usable_h = 0.86
    top_margin = 0.07

    node_pos: dict[str, tuple[float, float, float]] = {}
    for x, col in zip(col_x, columns):
        col_total = sum(node_value[n] for n in col)
        dynamic_gap = gap * max(0, len(col) - 1)
        scale = 0 if col_total == 0 else (usable_h - dynamic_gap) / col_total
        y = 1.0 - top_margin
        for n in col:
            h = node_value[n] * scale
            y0 = y - h
            node_pos[n] = (x, y0, y)
            y = y0 - gap

    node_colors = {
        "All Modules": "#264653",
        "Metadata clear STEM": "#457B9D",
        "Excluded non-STEM pattern": "#E76F51",
        "Enter semantic pipeline": "#6C757D",
        "Paragraph STEM override": "#2A9D8F",
        "Paragraph non-STEM guard": "#F4A261",
        "Enter sentence stage": "#A8DADC",
        "Sentence STEM override": "#3CB371",
        "Enter keyword fallback": "#A3B18A",
        "Keyword STEM override": "#7FB069",
        "Remain non-STEM or mixed": "#B8C0C8",
        "Final STEM": "#1B9E77",
        "Final non-STEM": "#D95F02",
    }

    for n, (x, y0, y1) in node_pos.items():
        color = node_colors.get(n, "#4f6fad")
        rect = Rectangle((x, y0), node_w, max(0.001, y1 - y0), facecolor=color, edgecolor="white", linewidth=0.6)
        ax.add_patch(rect)
        label_x = x + node_w + 0.004 if x < 0.5 else x - 0.004
        ha = "left" if x < 0.5 else "right"
        ax.text(label_x, (y0 + y1) / 2, f"{n}\n({node_value[n]})", va="center", ha=ha, fontsize=8.2, color="#1f2937")

    outgoing_offset = Counter()
    incoming_offset = Counter()

    def bezier_points(p0, p1, p2, p3, n=32):
        t = np.linspace(0, 1, n)
        return (
            ((1 - t) ** 3)[:, None] * p0
            + (3 * ((1 - t) ** 2) * t)[:, None] * p1
            + (3 * (1 - t) * (t**2))[:, None] * p2
            + (t**3)[:, None] * p3
        )

    link_color_by_source = {
        "All Modules": "#6C757D",
        "Enter semantic pipeline": "#6C757D",
        "Enter sentence stage": "#8AB17D",
        "Enter keyword fallback": "#B08968",
        "Metadata clear STEM": "#457B9D",
        "Paragraph STEM override": "#2A9D8F",
        "Sentence STEM override": "#3CB371",
        "Keyword STEM override": "#7FB069",
        "Excluded non-STEM pattern": "#E76F51",
        "Paragraph non-STEM guard": "#F4A261",
        "Remain non-STEM or mixed": "#9AA4AE",
    }

    for src, tgt, val in links:
        sx, sy0, sy1 = node_pos[src]
        tx, ty0, ty1 = node_pos[tgt]
        sh = max(0.0001, sy1 - sy0)
        th = max(0.0001, ty1 - ty0)

        sval = val / outflow[src] * sh if outflow[src] else 0
        tval = val / inflow[tgt] * th if inflow[tgt] else 0
        band_h = min(sval, tval)

        sy_top = sy0 + outgoing_offset[src] + band_h
        sy_bot = sy0 + outgoing_offset[src]
        outgoing_offset[src] += band_h

        ty_top = ty0 + incoming_offset[tgt] + band_h
        ty_bot = ty0 + incoming_offset[tgt]
        incoming_offset[tgt] += band_h

        x0 = sx + node_w
        x1 = tx
        cx = (x1 - x0) * 0.45

        top = bezier_points(np.array([x0, sy_top]), np.array([x0 + cx, sy_top]), np.array([x1 - cx, ty_top]), np.array([x1, ty_top]))
        bot = bezier_points(np.array([x1, ty_bot]), np.array([x1 - cx, ty_bot]), np.array([x0 + cx, sy_bot]), np.array([x0, sy_bot]))
        poly = np.vstack([top, bot])
        vertices = np.vstack([poly, poly[0]])
        codes = [Path.MOVETO] + [Path.LINETO] * (len(poly) - 1) + [Path.CLOSEPOLY]
        patch = PathPatch(
            Path(vertices, codes),
            facecolor=link_color_by_source.get(src, "#7f9ccf"),
            edgecolor="none",
            alpha=0.3,
        )
        ax.add_patch(patch)

    ax.text(0.02, 0.985, f"STEM Scope Classification Flow (N = {total})", ha="left", va="top", fontsize=13, color="#111827", weight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate STEM scope Sankey image (Matplotlib only).")
    parser.add_argument("--stem-input", type=Path, default=DEFAULT_STEM_INPUT)
    parser.add_argument("--non-stem-input", type=Path, default=DEFAULT_NON_STEM_INPUT)
    parser.add_argument("--image-output", type=Path, default=DEFAULT_IMAGE_OUTPUT)
    args = parser.parse_args()

    if not args.stem_input.exists():
        raise FileNotFoundError(f"Missing stem input: {args.stem_input}")
    if not args.non_stem_input.exists():
        raise FileNotFoundError(f"Missing non-stem input: {args.non_stem_input}")

    rows = load_jsonl(args.stem_input) + load_jsonl(args.non_stem_input)
    counts = build_counts(rows)

    render_image_with_matplotlib(counts=counts, output_path=args.image_output)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved Sankey image: {args.image_output}")


if __name__ == "__main__":
    main()
