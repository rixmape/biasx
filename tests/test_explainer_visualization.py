# test_explainer_visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pytest

from biasx.types import Gender, Box, FacialFeature
from biasx.explainers import Explainer, ClassActivationMapper
from biasx.models import Model


def visualize_explanation(pil_image, activation_map, activation_boxes, landmark_boxes, output_path):
    """
    Visualize the explanation results and save to an image file.
    
    Args:
        pil_image: Original PIL image
        activation_map: Heatmap of activations
        activation_boxes: Detected activation boxes
        landmark_boxes: Facial landmark boxes
        output_path: Path to save the visualization
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Explainer Visualization Test", fontsize=16)
    
    # Original image
    axes[0, 0].imshow(pil_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # Activation map
    axes[0, 1].imshow(pil_image)
    axes[0, 1].imshow(activation_map, alpha=0.6, cmap='jet')
    axes[0, 1].set_title("Activation Map")
    axes[0, 1].axis("off")
    
    # Activation boxes
    img_with_activation = pil_image.copy()
    draw = ImageDraw.Draw(img_with_activation)
    for box in activation_boxes:
        draw.rectangle(
            [(box.min_x, box.min_y), (box.max_x, box.max_y)],
            outline="red",
            width=2
        )
    axes[1, 0].imshow(img_with_activation)
    axes[1, 0].set_title("Activation Boxes")
    axes[1, 0].axis("off")
    
    # Landmark boxes with feature labels
    img_with_landmarks = pil_image.copy()
    draw = ImageDraw.Draw(img_with_landmarks)
    colors = {
        FacialFeature.LEFT_EYE: "blue",
        FacialFeature.RIGHT_EYE: "blue",
        FacialFeature.NOSE: "green",
        FacialFeature.LIPS: "yellow",
        FacialFeature.LEFT_CHEEK: "purple",
        FacialFeature.RIGHT_CHEEK: "purple",
        FacialFeature.LEFT_EYEBROW: "cyan",
        FacialFeature.RIGHT_EYEBROW: "cyan",
        FacialFeature.CHIN: "orange",
        FacialFeature.FOREHEAD: "pink",
    }
    
    for box in landmark_boxes:
        if box.feature:
            color = colors.get(box.feature, "white")
            draw.rectangle(
                [(box.min_x, box.min_y), (box.max_x, box.max_y)],
                outline=color,
                width=2
            )
            # Add feature label
            draw.text(
                (box.min_x, box.min_y - 10),
                box.feature.value,
                fill=color
            )
    
    axes[1, 1].imshow(img_with_landmarks)
    axes[1, 1].set_title("Facial Features")
    axes[1, 1].axis("off")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Visualization saved to {output_path}")


@pytest.mark.explainer
def test_explainer_visualization(mock_config, sample_image, mock_tf_model, mock_landmarker, mock_cam, tmpdir):
    """
    Test visualization of explainer results.
    """
    # Create output directory
    output_dir = os.path.join(tmpdir, "explainer_output")
    os.makedirs(output_dir, exist_ok=True)

    # Set up the explainer with mocked components
    explainer = Explainer(
        landmarker_source="mediapipe",
        cam_method="gradcam++",
        cutoff_percentile=90,
        threshold_method="otsu",
        overlap_threshold=0.2,
        distance_metric="euclidean",
        batch_size=1
    )

    # Replace the real components with mocks
    explainer.landmarker = mock_landmarker.return_value
    explainer.activation_mapper = mock_cam.return_value

    # Create a preprocessed image
    preprocessed_image = np.array(sample_image.convert('L')) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)

    # Create a mock model instead of trying to load a real one
    with patch('biasx.models.Model') as MockModel:
        # Create a mock model instance
        mock_model = MockModel.return_value
        mock_model.model = mock_tf_model
        mock_model.inverted_classes = False
        mock_model.batch_size = 1
        
        # Run the explainer with the mocked model
        activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
            pil_images=[sample_image],
            preprocessed_images=[preprocessed_image],
            model=mock_model,
            target_classes=[Gender.MALE]
        )
        
        # Generate visualization
        output_path = os.path.join(output_dir, "explainer_visualization.png")
        visualize_explanation(
            sample_image,
            activation_maps[0],
            activation_boxes[0],
            landmark_boxes[0],
            output_path
        )
        
        # Simple assertions to make sure the test ran
        assert len(activation_maps) == 1
        assert len(activation_boxes) == 1
        assert len(landmark_boxes) == 1
        assert os.path.exists(output_path)
        
        print(f"\nVisualization saved to: {output_path}")

@pytest.mark.explainer
def test_real_explainer_with_sample_face(tmpdir):
    """
    Test visualization with a real face image (if available).
    
    This test attempts to load a real face image and run the actual explainer
    to generate a visualization.
    """
    try:
        # Try to find a test image
        sample_dir = os.path.join(os.path.dirname(__file__), "data/sample_images")
        if not os.path.exists(sample_dir):
            pytest.skip("Sample images directory not found")
            
        # Try to find a face image
        image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
        if not image_files:
            pytest.skip("No sample images found")
            
        # Load the first image
        image_path = os.path.join(sample_dir, image_files[0])
        image = Image.open(image_path)
        
        # Create output directory
        output_dir = os.path.join(tmpdir, "real_explainer_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a simple test model
        from biasx.models import Model
        model_path = os.path.join(os.path.dirname(__file__), "data/sample_models/test_model.h5")
        if not os.path.exists(model_path):
            pytest.skip("Test model not found")
            
        model = Model(path=model_path, inverted_classes=False, batch_size=1)
        
        # Create the explainer
        explainer = Explainer(
            landmarker_source="mediapipe",
            cam_method="gradcam++",
            cutoff_percentile=90,
            threshold_method="otsu",
            overlap_threshold=0.2,
            distance_metric="euclidean",
            batch_size=1
        )
        
        # Preprocess the image
        preprocessed_image = np.array(image.convert('L').resize((48, 48))) / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
        
        # Run the explainer
        activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
            pil_images=[image],
            preprocessed_images=[preprocessed_image],
            model=model,
            target_classes=[Gender.MALE]
        )
        
        # Generate visualization
        output_path = os.path.join(output_dir, "real_explainer_visualization.png")
        visualize_explanation(
            image,
            activation_maps[0],
            activation_boxes[0],
            landmark_boxes[0],
            output_path
        )
        
        assert os.path.exists(output_path)
        
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Required dependencies not available")
    except Exception as e:
        pytest.skip(f"Failed to run real explainer: {e}")