"""Arrow, scatter, and spectrum visualizations for strategy-space."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap.umap_ as umap
from plotly.figure_factory import create_quiver

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_EMBEDDINGS = ROOT / "outputs" / "embeddings"
DEFAULT_OUTPUT = ROOT / "outputs" / "visualizations"
SP500_FILE = DATA / "SP500.csv"
AXIS_FILE = DATA / "axis_definition.csv"

YEAR = "2025"
Q_START = "1"
Q_END = "3"
BINS = 5
QUARTERS = ("1", "2", "3", "4")

EXCLUDED = {
    "BF.B", "BRK.B", "DELL", "ED", "EXPD", "FOX", "GOOG",
    "MPWR", "NVR", "NWS", "PSKY", "Q", "SCHW",
}

MAPPING = {"BFb": None, "FOXA": "FOXA", "GOOGL": "GOOGL", "NWSA": "NWSA"}

SECTOR_COLORS = {
    "Industrials": "#1f77b4",
    "Materials": "#aec7e8",
    "Financials": "#2ca02c",
    "Real Estate": "#98df8a",
    "Information Technology": "#9467bd",
    "Communication Services": "#c5b0d5",
    "Health Care": "#d62728",
    "Consumer Discretionary": "#ff7f0e",
    "Consumer Staples": "#ffbb78",
    "Utilities": "#8c564b",
    "Energy": "#e377c2",
}

SECTOR_ORDER = [
    "Industrials", "Materials", "Energy", "Utilities",
    "Consumer Discretionary", "Consumer Staples",
    "Information Technology", "Communication Services",
    "Financials", "Real Estate", "Health Care",
]

MAJOR = {
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "JPM", "BAC", "GS",
    "MS", "WFC", "V", "MA", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS",
    "COST", "JNJ", "UNH", "PFE", "ABBV", "TMO", "LLY", "BA", "CAT",
    "GE", "MMM", "HON", "XOM", "CVX", "COP", "BRK.A", "TSMC", "KO",
    "PEP", "PG",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def load_sp500():
    df = pd.read_csv(SP500_FILE, encoding="utf-8-sig")
    df = df[~df["Symbol"].isin(EXCLUDED)].copy()
    return df.set_index("Symbol")[["GICS_sector", "Security"]]


def load_embedding_jsons(folder):
    data = {}
    for path in Path(folder).glob("*.json"):
        ticker = MAPPING.get(path.stem, path.stem)
        if ticker is None:
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data[ticker] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    if not data:
        raise FileNotFoundError(f"No embedding JSON files found in: {folder}")
    return data


def quarter_embedding(data, quarter):
    try:
        value = data[YEAR][quarter]["embedding"]
        return np.asarray(value, dtype=np.float32) if value else None
    except (KeyError, TypeError):
        return None


def year_embedding(data, mode="quarterly"):
    year = data.get(YEAR, {})
    if mode == "year" and isinstance(year, dict) and year.get("embedding"):
        return np.asarray(year["embedding"], dtype=np.float32)
    vectors = [quarter_embedding(data, q) for q in QUARTERS]
    vectors = [v for v in vectors if v is not None]
    return np.mean(vectors, axis=0) if vectors else None


def parse_vector(value):
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def anchors_from_json(path):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("anchors", raw) if isinstance(raw, dict) else raw
    anchors = {}
    iterable = items.items() if isinstance(items, dict) else [(x.get("name"), x) for x in items]
    for name, item in iterable:
        if not name or not isinstance(item, dict) or item.get("type", "bipolar") != "bipolar":
            continue
        if "vector" in item:
            vec = parse_vector(item["vector"])
        elif item.get("pos_definitions") and item.get("neg_definitions"):
            pos = np.mean([parse_vector(x["vector"]) for x in item["pos_definitions"]], axis=0)
            neg = np.mean([parse_vector(x["vector"]) for x in item["neg_definitions"]], axis=0)
            vec = pos - neg
        else:
            continue
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        parts = name.split("_")
        anchors[name] = {
            "vector": vec / norm,
            "pos_label": item.get("pos_label", parts[0].capitalize()),
            "neg_label": item.get("neg_label", parts[1].capitalize() if len(parts) > 1 else "Opposite"),
        }
    return anchors


def anchors_from_csv(path):
    df = pd.read_csv(path)
    name_col = next((c for c in ("name", "axis_name", "axis") if c in df.columns), None)
    if not name_col:
        return {}
    anchors = {}
    for _, row in df.iterrows():
        name = str(row[name_col])
        try:
            if "vector" in df.columns and pd.notna(row["vector"]):
                vec = parse_vector(row["vector"])
            elif {"pos_vector", "neg_vector"}.issubset(df.columns):
                vec = parse_vector(row["pos_vector"]) - parse_vector(row["neg_vector"])
            else:
                continue
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            parts = name.split("_")
            anchors[name] = {
                "vector": vec / norm,
                "pos_label": str(row.get("pos_label", parts[0].capitalize())),
                "neg_label": str(row.get("neg_label", parts[1].capitalize() if len(parts) > 1 else "Opposite")),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return anchors


def load_anchors(anchor_path=None):
    candidates = []
    if anchor_path:
        candidates.append(Path(anchor_path))
    candidates += [
        DATA / "anchors_embedded.json",
        DATA / "anchors.json",
        ROOT / "outputs" / "anchors_embedded.json",
        ROOT / "outputs" / "anchors.json",
        AXIS_FILE,
    ]
    for path in candidates:
        if not path.exists():
            continue
        anchors = anchors_from_csv(path) if path.suffix.lower() == ".csv" else anchors_from_json(path)
        if anchors:
            return anchors
    raise FileNotFoundError(
        "No anchor vectors found. Run 3_create_anchors.py first, or pass --anchors. "
        "axis_definition.csv can be read directly only when it contains vector columns."
    )


def project_companies(raw, meta, anchors, mode="quarterly"):
    rows = []
    for ticker, data in raw.items():
        if ticker not in meta.index:
            continue
        emb = year_embedding(data, mode)
        if emb is None:
            continue
        row = {
            "ticker": ticker,
            "sector": meta.at[ticker, "GICS_sector"],
            "security": meta.at[ticker, "Security"],
        }
        for name, info in anchors.items():
            denom = np.linalg.norm(emb) * np.linalg.norm(info["vector"])
            row[name] = float(np.dot(emb, info["vector"]) / denom) if denom else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def movement_frame(raw, meta):
    rows, vectors = [], []
    for ticker, data in raw.items():
        if ticker not in meta.index:
            continue
        start, end = quarter_embedding(data, Q_START), quarter_embedding(data, Q_END)
        if start is None or end is None:
            continue
        denom = np.linalg.norm(start) * np.linalg.norm(end)
        distance = 1 - float(np.dot(start, end) / denom) if denom else 0.0
        idx = len(vectors)
        vectors.extend([start, end])
        rows.append((ticker, idx, distance))
    if not rows:
        return pd.DataFrame()
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, metric="cosine").fit_transform(np.asarray(vectors))
    out = []
    for ticker, idx, distance in rows:
        out.append({
            "ticker": ticker,
            "security": meta.at[ticker, "Security"],
            "sector": meta.at[ticker, "GICS_sector"],
            "x1": coords[idx, 0], "y1": coords[idx, 1],
            "x2": coords[idx + 1, 0], "y2": coords[idx + 1, 1],
            "distance": distance,
        })
    return pd.DataFrame(out)


def xy_range(df, pad=0.05):
    xs = np.r_[df.x1, df.x2]
    ys = np.r_[df.y1, df.y2]
    dx, dy = np.ptp(xs) or 1.0, np.ptp(ys) or 1.0
    return (xs.min() - pad * dx, xs.max() + pad * dx), (ys.min() - pad * dy, ys.max() + pad * dy)


def plot_arrows(df, out):
    if df.empty:
        return
    (x0, x1), (y0, y1) = xy_range(df)
    dmin, dmax = df.distance.min(), df.distance.max()
    den = dmax - dmin or 1.0

    fig, ax = plt.subplots(figsize=(24, 16))
    for _, r in df.iterrows():
        lw = 2.0 + ((r.distance - dmin) / den) * 5.0
        ax.annotate("", (r.x2, r.y2), (r.x1, r.y1), arrowprops={
            "arrowstyle": "-|>", "color": SECTOR_COLORS.get(r.sector, "#7f7f7f"),
            "alpha": 0.6, "lw": lw, "mutation_scale": 35, "shrinkA": 0, "shrinkB": 0,
        })
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[], yticks=[])
    ax.axis("off")
    fig.savefig(out / f"arrow_{YEAR}_Q{Q_START}_Q{Q_END}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig = go.Figure()
    edges = np.quantile(df.distance, np.linspace(0, 1, BINS + 1)) if not np.allclose(dmin, dmax) else np.array([dmin, dmax + 1e-9])
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bins = len(edges) - 1
    for sector, sdf in df.groupby("sector"):
        color = SECTOR_COLORS.get(sector, "#7f7f7f")
        for bi in range(bins):
            sub = sdf[(sdf.distance >= edges[bi]) & (sdf.distance < edges[bi + 1])]
            if sub.empty:
                continue
            q = create_quiver(
                sub.x1, sub.y1, sub.x2 - sub.x1, sub.y2 - sub.y1,
                scale=1, arrow_scale=0.7, line_color=color, line_width=3.0 + bi * 4.0,
            )
            for tr in q.data:
                tr.opacity = 0.4 + bi / max(bins - 1, 1) * 0.45
                tr.showlegend = False
                tr.hoverinfo = "skip"
                fig.add_trace(tr)
        fig.add_trace(go.Scatter(
            x=(sdf.x1 + sdf.x2) / 2, y=(sdf.y1 + sdf.y2) / 2,
            mode="markers", marker={"size": 10, "color": color, "opacity": 0},
            text=sdf.ticker, customdata=sdf[["security", "sector", "distance"]],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<br>Distance: %{customdata[2]:.6f}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white", plot_bgcolor="white")
    fig.update_xaxes(visible=False, range=[x0, x1])
    fig.update_yaxes(visible=False, range=[y0, y1], scaleanchor="x", scaleratio=1)
    fig.write_html(out / f"arrow_{YEAR}_Q{Q_START}_Q{Q_END}.html", include_plotlyjs="cdn")


def plot_scatter(df, axes, out, suffix):
    names = list(axes)[:2]
    if len(names) < 2 or df.empty:
        return
    x_name, y_name = names
    fig = go.Figure()
    for sector, sdf in df.groupby("sector"):
        fig.add_trace(go.Scatter(
            x=sdf[x_name], y=sdf[y_name], mode="markers", name=sector,
            marker={"size": 9, "color": SECTOR_COLORS.get(sector, "#7f7f7f"), "opacity": 0.75},
            text=sdf.ticker, customdata=sdf[["security"]],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
        ))
    fig.update_layout(template="plotly_white", xaxis_title=x_name, yaxis_title=y_name)
    fig.write_html(out / f"scatter_{suffix}.html", include_plotlyjs="cdn")


def spectrum_bounds(df, axis):
    m = max(abs(df[axis].min()), abs(df[axis].max())) or 1.0
    return -1.1 * m, 1.1 * m


def plot_spectrum(df, axes, out, suffix):
    for axis, info in axes.items():
        xmin, xmax = spectrum_bounds(df, axis)
        fig, ax = plt.subplots(figsize=(14, 10))
        for i, sector in enumerate(SECTOR_ORDER):
            sdf = df[df.sector == sector]
            if sdf.empty:
                continue
            y = len(SECTOR_ORDER) - i - 1
            color = SECTOR_COLORS.get(sector, "#7f7f7f")
            ax.add_patch(plt.Rectangle((xmin, y - 0.35), xmax - xmin, 0.7, fill=False, lw=1))
            ax.vlines(sdf[axis], y - 0.35, y + 0.35, color=color, alpha=0.7, lw=0.8)
            ax.vlines(sdf[axis].mean(), y - 0.35, y + 0.35, color="black", lw=2.5)
            ax.text(xmin - (xmax - xmin) * 0.03, y, sector, ha="right", va="center", fontsize=9)
        ax.axvline(0, color="gray", ls="--", alpha=0.5)
        ax.set(xlim=(xmin, xmax), ylim=(-0.5, len(SECTOR_ORDER) - 0.5), yticks=[], xlabel="Projection Score")
        ax.set_title(axis.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.2, ls="--")
        fig.savefig(out / f"spectrum_{suffix}_{axis}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        pfig = go.Figure()
        for i, sector in enumerate(SECTOR_ORDER):
            sdf = df[df.sector == sector]
            if sdf.empty:
                continue
            y = len(SECTOR_ORDER) - i - 1
            color = SECTOR_COLORS.get(sector, "#7f7f7f")
            for _, r in sdf.iterrows():
                pfig.add_trace(go.Scatter(
                    x=[r[axis], r[axis]], y=[y - 0.35, y + 0.35], mode="lines",
                    line={"color": color, "width": 2}, text=[r.ticker, r.ticker],
                    hovertemplate=f"<b>{r.ticker}</b><br>{r.security}<br>Score: {r[axis]:.4f}<extra></extra>",
                    showlegend=False,
                ))
            mean = sdf[axis].mean()
            pfig.add_trace(go.Scatter(x=[mean, mean], y=[y - 0.35, y + 0.35], mode="lines", line={"color": "black", "width": 4}, showlegend=False))
        pfig.add_vline(x=0, line_dash="dash", line_color="gray")
        pfig.update_layout(
            template="plotly_white", height=800,
            title=f"{axis.replace('_', ' ').title()}<br><sub>{info['neg_label']} ←→ {info['pos_label']}</sub>",
            xaxis={"title": "Projection Score", "range": [xmin, xmax]},
            yaxis={"tickmode": "array", "tickvals": list(range(len(SECTOR_ORDER))), "ticktext": list(reversed(SECTOR_ORDER))},
        )
        pfig.write_html(out / f"spectrum_{suffix}_{axis}.html", include_plotlyjs="cdn")


def save_scores(df, axes, out, suffix):
    cols = list(axes)
    df.to_csv(out / f"company_scores_{suffix}.csv", index=False, encoding="utf-8-sig")
    df.groupby("sector")[cols].agg(["mean", "std", "count"]).round(4).to_csv(
        out / f"sector_summary_{suffix}.csv", encoding="utf-8-sig"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    meta = load_sp500()
    raw = load_embedding_jsons(args.embeddings)
    anchors = load_anchors(args.anchors)

    movement = movement_frame(raw, meta)
    plot_arrows(movement, args.output)
    movement.to_csv(args.output / f"movement_{YEAR}_Q{Q_START}_Q{Q_END}.csv", index=False)

    quarterly = project_companies(raw, meta, anchors, "quarterly")
    plot_scatter(quarterly, anchors, args.output, "quarterly_avg")
    plot_spectrum(quarterly, anchors, args.output, "quarterly_avg")
    save_scores(quarterly, anchors, args.output, "quarterly_avg")

    yearly = project_companies(raw, meta, anchors, "year")
    if not yearly.empty:
        plot_scatter(yearly, anchors, args.output, "year")
        plot_spectrum(yearly, anchors, args.output, "year")
        save_scores(yearly, anchors, args.output, "year")


if __name__ == "__main__":
    main()
