import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass
class FacialRegionBox:
    """Represents a bounding box around a specific facial feature."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    name: str
    importance: Optional[float] = None


def plot_dataset_distribution(data: pd.DataFrame, path: str = "dataset_distribution.png") -> None:
    """Generates and saves a visualization of gender, race, and age group distributions from the given dataset."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    gender_counts = data["gender"].value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    gender_counts["Gender"] = gender_counts["Gender"].map({0: "Male", 1: "Female"})
    sns.histplot(data=gender_counts, x="Gender", weights="Count", discrete=True, color="steelblue")
    plt.title("Gender Distribution")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    race_counts = data["race"].value_counts().reset_index()
    race_counts.columns = ["Race", "Count"]
    race_mapping = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
    race_counts["Race"] = race_counts["Race"].map(race_mapping)
    sns.histplot(data=race_counts, x="Race", weights="Count", discrete=True, color="steelblue")
    plt.title("Race Distribution")
    plt.xticks()
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    age_mapping = {0: "0-9", 1: "10-19", 2: "20-29", 3: "30-39", 4: "40-49", 5: "50-59", 6: "60-69", 7: "70+"}
    age_order = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    data["age_group"] = pd.Categorical(data["age"].map(age_mapping), categories=age_order, ordered=True)
    sns.histplot(data["age_group"], discrete=True, kde=False, color="steelblue")
    plt.title("Age Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_image_grid(images: np.ndarray, labels: Optional[np.ndarray] = None, predictions: Optional[np.ndarray] = None, heatmaps: Optional[list[np.ndarray]] = None, boxes: Optional[list[list[FacialRegionBox]]] = None, max_images: int = 32, cmap: str = "gray", path: str = "image_grid.png") -> None:
    """Plots a grid of images with optional labels, predictions, heatmaps, and bounding boxes, and saves the result to a file."""
    num_images = min(max_images, len(images))
    cols = min(8, num_images)
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        img = images[i].reshape(images[i].shape[:2])
        plt.imshow(img, cmap=cmap)

        if heatmaps is not None:
            plt.imshow(heatmaps[i], cmap="jet", alpha=0.5)

        if boxes is not None and boxes[i]:
            for box in boxes[i]:
                rect = plt.Rectangle((box.min_x, box.min_y), box.max_x - box.min_x, box.max_y - box.min_y, linewidth=1, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                if hasattr(box, "importance") and box.importance is not None:
                    plt.text(box.min_x, box.min_y - 2, f"{box.name}: {box.importance:.2f}", color="red", fontsize=8, backgroundcolor="white")

        if labels is not None:
            title = f"True: {'Male' if labels[i] == 0 else 'Female'}"
            if predictions is not None:
                title += f"\nPred: {'Male' if predictions[i] == 0 else 'Female'}"
                color = "green" if predictions[i] == labels[i] else "red"
                plt.title(title, color=color)
            else:
                plt.title(title)

        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_training_history(history: dict[str, list[float]], path: str = "training_history.png") -> None:
    """Generates and saves a plot of the training and validation accuracy and loss over epochs."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", marker="x")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training Loss", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", marker="x")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str = "confusion_matrix.png") -> None:
    """Generates and saves a confusion matrix plot comparing true and predicted labels."""
    labels = ["Male", "Female"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_probabilities(feature_parities: dict[str, dict[str, float]], path: str = "feature_probabilities.png") -> None:
    """Plots and saves a bar chart comparing male and female importance probabilities for facial features, annotated with bias values."""
    features = list(feature_parities.keys())
    male_probs = [feature_parities[r]["male"] for r in features]
    female_probs = [feature_parities[r]["female"] for r in features]
    biases = [feature_parities[r]["bias"] for r in features]

    sorted_indices = np.argsort(biases)[::-1]
    features = [features[i] for i in sorted_indices]
    male_probs = [male_probs[i] for i in sorted_indices]
    female_probs = [female_probs[i] for i in sorted_indices]
    biases = [biases[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 1, 1)
    x = np.arange(len(features))
    width = 0.35

    ax1.bar(x - width / 2, male_probs, width, label="Male", color="skyblue")
    ax1.bar(x + width / 2, female_probs, width, label="Female", color="pink")

    ax1.set_xlabel("Facial Region")
    ax1.set_ylabel("Importance Probability")
    ax1.set_title("Facial Region Importance by Gender")
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45, ha="right")
    ax1.legend()

    ax1.set_ylim(0, 1.1)

    for i, (m, f, b) in enumerate(zip(male_probs, female_probs, biases)):
        ax1.text(i, max(m, f) + 0.02, f"Bias: {b:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bias_scores(scores: dict[str, float], path: str = "bias_scores.png") -> None:
    """Generates and saves a horizontal bar chart of bias scores."""
    plt.figure(figsize=(8, 6))

    bars = plt.barh(list(scores.keys()), list(scores.values()), color="steelblue")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center")

    plt.xlabel("Score Value")
    plt.title("Bias Metrics")
    plt.xlim(0, 1.1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
