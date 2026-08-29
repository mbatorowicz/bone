"""Wykresy ewolucji i rzut pozycji."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bone.traits import Universe


def save_timeseries(history: list[dict[str, float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(df["t"], df["mean_knowledge"], label="wiedza")
    ax.plot(df["t"], df["mean_wisdom"], label="madrosc")
    ax.plot(df["t"], df["mean_health"], label="zdrowie")
    ax.set_title("Wiedza / madrosc / zdrowie")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df["t"], df["mean_love"], label="milosc")
    ax.plot(df["t"], df["mean_hatred"], label="nienawisc")
    ax.plot(df["t"], df["mean_loyalty"], label="lojalnosc")
    ax.plot(df["t"], df["mean_anger"], label="zlosc")
    ax.set_title("Emocje spoleczne")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    if "mean_wealth" in df.columns:
        ax.plot(df["t"], df["mean_wealth"], label="srednie bogactwo")
        ax.plot(df["t"], df["max_wealth"], label="max bogactwo")
        ax.plot(df["t"], df["gini"], label="Gini")
        ax.plot(df["t"], df["money_velocity"], label="predkosc pieniadza")
    ax.set_title("Przeplyw dobr / pieniadze")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(df["t"], df["n_clusters"], label="klastry >= min")
    ax.plot(df["t"], df["largest_cluster"], label="najwiekszy klaster")
    ax.plot(df["t"], df["polarization"], label="polaryzacja")
    ax.set_title("Struktura spoleczna")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(df["t"], df["mean_speed"], label="srednia |v|")
    ax.plot(df["t"], df["mean_gamma"], label="srednia gamma")
    ax.plot(df["t"], df["n_alive"], label="zywi")
    ax.set_title("Kinematyka i przezycie")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    if "flow_traded" in df.columns:
        ax.plot(df["t"], df["flow_traded"], label="handel")
        ax.plot(df["t"], df["flow_exploited"], label="wyzysk")
        ax.plot(df["t"], df["flow_gifted"], label="dary")
    ax.set_title("Sklad przeplywow (krok)")
    ax.set_xlabel("czas t")
    ax.legend(fontsize=8)

    fig_path = out_dir / "timeseries.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    return fig_path


def save_scatter_projection(universe: Universe, out_dir: Path, tag: str = "final") -> Path:
    """Rzut XY: kolor = bogactwo, rozmiar ~ wytrwalosc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    alive = universe.alive_mask
    pos = universe.positions[alive]
    wealth = universe.traits["wealth"][alive]
    endurance = universe.traits["endurance"][alive]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=wealth,
        s=8 + 40 * endurance,
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("bogactwo (wealth)")
    ax.set_aspect("equal")
    ax.set_title(f"Rzut XY wealth ({tag}), t={universe.t:.2f}, n={int(alive.sum())}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    path = out_dir / f"scatter_xy_{tag}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_trait_histograms(universe: Universe, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    alive = universe.alive_mask
    names = ["knowledge", "wisdom", "love", "hatred", "loyalty", "wealth"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ax, name in zip(axes.ravel(), names):
        ax.hist(universe.traits[name][alive], bins=30, color="#3d5a80", alpha=0.85)
        ax.set_title(name)
    path = out_dir / "trait_histograms.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
