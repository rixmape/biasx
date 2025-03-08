import pandas as pd
from matplotlib import pyplot as plt

DEMOGRAPHIC_COLUMNS: list[str] = ["gender", "race", "age"]


def get_demographic_distribution(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Calculate percentage distribution of demographic attributes."""
    distribution = {}
    for col in DEMOGRAPHIC_COLUMNS:
        counts = df[col].value_counts(normalize=True) * 100
        distribution[col] = {int(k): float(v) for k, v in counts.items()}
    return distribution


def create_image_grid(df: pd.DataFrame, output_path: str, rows: int = 4, cols: int = 5, dpi: int = 300, seed: int = None) -> str:
    """Generate a grid visualization of sample images with optional demographic labels."""
    sample_size = min(rows * cols, len(df))
    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes_flat:
        ax.axis("off")
    for idx in range(sample_size):
        row = sampled.iloc[idx]
        axes_flat[idx].imshow(row["image"])
        label_text = ", ".join(f"{col[0].upper()}: {row[col]}" for col in DEMOGRAPHIC_COLUMNS)
        axes_flat[idx].set_title(label_text, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
