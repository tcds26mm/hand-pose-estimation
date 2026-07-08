"""Analyze Ultralytics training results.csv and plot loss curves for the lab report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path("runs/pose/train/results.csv")
DEFAULT_OUT = Path("runs/pose/train/training_curves.png")


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"results.csv not found at {csv_path}\n"
            "Unzip your Kaggle runs.zip so it lives at runs/pose/train/results.csv"
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def print_report(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Print metrics for README.md and return the best-epoch snapshot."""
    epoch_col = pick_column(df, ["epoch"])
    if epoch_col is None:
        raise ValueError("No 'epoch' column found in results.csv")

    # Pose mAP columns (Ultralytics pose runs use (P) suffix)
    map50_col = pick_column(
        df,
        ["metrics/mAP50(P)", "metrics/mAP50(B)", "metrics/mAP50"],
    )
    map5095_col = pick_column(
        df,
        ["metrics/mAP50-95(P)", "metrics/mAP50-95(B)", "metrics/mAP50-95"],
    )

    val_pose_col = pick_column(df, ["val/pose_loss"])
    val_box_col = pick_column(df, ["val/box_loss"])
    val_cls_col = pick_column(df, ["val/cls_loss"])

    if map50_col:
        best_idx = df[map50_col].idxmax()
    elif val_pose_col:
        best_idx = df[val_pose_col].idxmin()
    else:
        best_idx = len(df) - 1

    best = df.loc[best_idx]
    epochs_done = int(df[epoch_col].iloc[-1])
    time_col = pick_column(df, ["time"])

    report = {
        "epochs_completed": epochs_done,
        "best_epoch": int(best[epoch_col]),
        "box_loss": float(best[val_box_col]) if val_box_col else None,
        "pose_loss": float(best[val_pose_col]) if val_pose_col else None,
        "cls_loss": float(best[val_cls_col]) if val_cls_col else None,
        "pose_mAP50": float(best[map50_col]) if map50_col else None,
        "pose_mAP50_95": float(best[map5095_col]) if map5095_col else None,
    }

    if time_col and len(df) > 1:
        per_epoch_sec = df[time_col].diff().dropna().mean()
        report["avg_seconds_per_epoch"] = float(per_epoch_sec)

    print("=" * 60)
    print("TRAINING REPORT (paste into README.md)")
    print("=" * 60)
    print(f"Overall Training Budget Epochs Completed: {report['epochs_completed']}")
    if "avg_seconds_per_epoch" in report:
        m, s = divmod(int(report["avg_seconds_per_epoch"]), 60)
        print(f"Absolute Processing Duration Per Epoch: {m} min {s} sec (avg)")
    print(f"Best checkpoint epoch (by primary metric): {report['best_epoch']}")
    if report["box_loss"] is not None:
        print(f"Box Loss (val/box_loss): {report['box_loss']:.5f}")
    if report["pose_loss"] is not None:
        print(f"Pose Loss (val/pose_loss): {report['pose_loss']:.5f}")
    if report["cls_loss"] is not None:
        print(f"Class Loss (val/cls_loss): {report['cls_loss']:.5f}")
    if report["pose_mAP50"] is not None:
        print(f"Tracking Precision Score (Pose mAP50): {report['pose_mAP50']:.5f}")
    if report["pose_mAP50_95"] is not None:
        print(f"Rigorous Generalization Bound (Pose mAP50-95): {report['pose_mAP50_95']:.5f}")
    print("=" * 60)

    return report


def plot_loss_curves(df: pd.DataFrame, out_path: Path) -> None:
    """Plot train vs validation loss curves."""
    epoch_col = pick_column(df, ["epoch"])
    if epoch_col is None:
        raise ValueError("No 'epoch' column found in results.csv")

    loss_groups = [
        ("Box loss", "train/box_loss", "val/box_loss"),
        ("Pose loss", "train/pose_loss", "val/pose_loss"),
        ("Class loss", "train/cls_loss", "val/cls_loss"),
    ]

    available = [
        (title, train_col, val_col)
        for title, train_col, val_col in loss_groups
        if train_col in df.columns or val_col in df.columns
    ]
    if not available:
        raise ValueError("No loss columns found in results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (title, train_col, val_col) in zip(axes, available):
        if train_col in df.columns:
            ax.plot(df[epoch_col], df[train_col], label="train", linewidth=2)
        if val_col in df.columns:
            ax.plot(df[epoch_col], df[val_col], label="val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Train vs Validation Loss Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YOLO training results.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to results.csv (default: runs/pose/train/results.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output plot path (default: runs/pose/train/training_curves.png)",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    print_report(df)
    plot_loss_curves(df, args.out)


if __name__ == "__main__":
    main()
