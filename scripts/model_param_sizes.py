#!/usr/bin/env python
"""
Script to print parameter sizes and feature dimensions for all models in LazySlide.

This script uses rich to display parameter sizes in a nice table with human-readable format.
For vision and multimodal models, it also displays the output feature dimension from encode_image.

Usage:
    python model_param_sizes.py [options]

Options:
    --task TASK            Filter models by task (vision, segmentation, multimodal, tile_prediction)
    --models MODEL [MODEL ...]  Specify one or more model names to display (partial matching supported)
    --skip-load            Skip loading models (will only show metadata)
    --update-registry      Update model_registry.json with calculated parameter sizes and feature dimensions
    --detailed             Show detailed information including license and commercial status
    --show-features        Show output feature dimensions for vision and multimodal models (default: True)
    --no-features          Don't show output feature dimensions

Examples:
    # Show parameter sizes for all models (skipping loading)
    python model_param_sizes.py --skip-load

    # Show detailed information for vision models only
    python model_param_sizes.py --task vision --detailed --skip-load

    # Calculate parameter sizes and feature dimensions for non-gated models and update the registry
    python model_param_sizes.py --update-registry

    # Show only parameter sizes without feature dimensions
    python model_param_sizes.py --no-features

    # Show information for specific models only
    python model_param_sizes.py --models vit resnet

    # Combine with other options
    python model_param_sizes.py --models clip --detailed --skip-load
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

import lazyslide as zs
from lazyslide.models._model_registry import MODEL_DB, MODEL_REGISTRY, ModelTask
from lazyslide.models.base import ImageModel, ImageTextModel


def human_readable_size(num_params):
    """Convert number of parameters to human-readable format."""
    if num_params < 1e3:
        return f"{num_params}"
    elif num_params < 1e6:
        return f"{num_params / 1e3:.1f}K"
    elif num_params < 1e9:
        return f"{num_params / 1e6:.1f}M"
    else:
        return f"{num_params / 1e9:.2f}B"


def count_parameters(model):
    """Count the number of parameters in a model."""
    if not isinstance(model, torch.nn.Module):
        try:
            # If it's a LazySlide model, get the underlying PyTorch model
            model = model.model
        except (AttributeError, TypeError):
            return 0

    return sum(p.numel() for p in model.parameters())


def determine_feature_dimension(model_instance):
    """Determine the output feature dimension for a model.

    This function creates a dummy input and runs encode_image on it to get the output shape.

    Parameters
    ----------
    model_instance : ImageModel or ImageTextModel
        The model instance to determine the feature dimension for

    Returns
    -------
    int or None
        The feature dimension if it can be determined, None otherwise
    """
    if not isinstance(model_instance, (ImageModel, ImageTextModel)):
        return None

    try:
        # Create a dummy input (batch size 1, 3 channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Get the transform if available
        transform = model_instance.get_transform()
        if transform is not None:
            dummy_input = transform(dummy_input)

        # Run encode_image on the dummy input
        with torch.no_grad():
            output = model_instance.encode_image(dummy_input)

        # Determine the feature dimension
        if isinstance(output, torch.Tensor):
            # If output is a tensor, get the last dimension
            return output.shape[-1]
        elif isinstance(output, np.ndarray):
            # If output is a numpy array, get the last dimension
            return output.shape[-1]
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            # If output is a list or tuple, get the last dimension of the first element
            first_item = output[0]
            if isinstance(first_item, (torch.Tensor, np.ndarray)):
                return first_item.shape[-1]

        return None
    except Exception as e:
        print(f"Error determining feature dimension: {str(e)}")
        return None


def get_model_info_from_registry(model_name):
    """Get model information from the registry."""
    for model_info in MODEL_DB:
        if model_info["name"] == model_name:
            return model_info
    return None


def update_model_registry(model_name, param_size=None, feature_dim=None):
    """Update the model registry with parameter size and feature dimension information.

    Parameters
    ----------
    model_name : str
        The name of the model to update
    param_size : str, optional
        The parameter size in human-readable format (e.g., "2.1M")
    feature_dim : int, optional
        The output feature dimension from encode_image

    Returns
    -------
    bool
        True if the registry was updated, False otherwise
    """
    registry_path = (
        Path(__file__).parent / "src" / "lazyslide" / "models" / "model_registry.json"
    )

    if not registry_path.exists():
        # Try to find it relative to the current directory
        registry_path = Path("src") / "lazyslide" / "models" / "model_registry.json"
        if not registry_path.exists():
            print(f"Could not find model_registry.json at {registry_path}")
            return False

    # Read the current registry
    with open(registry_path, "r") as f:
        registry = json.load(f)

    # Update the parameter size and feature dimension for the specified model
    updated = False
    for model in registry:
        if model["name"] == model_name:
            if param_size is not None:
                model["param_size"] = param_size
                updated = True

            if feature_dim is not None:
                model["encode_dim"] = feature_dim
                updated = True

            break

    if updated:
        # Write the updated registry back to the file
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Print parameter sizes and feature dimensions for LazySlide models"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[t.value for t in ModelTask],
        help="Filter models by task",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specify one or more model names to display",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading models (will only show metadata)",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Update model_registry.json with calculated parameter sizes and feature dimensions",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information including license and commercial status",
    )

    # Add mutually exclusive group for feature dimension display options
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--show-features",
        action="store_true",
        default=True,
        help="Show output feature dimensions for vision and multimodal models (default)",
    )
    feature_group.add_argument(
        "--no-features",
        action="store_true",
        help="Don't show output feature dimensions",
    )

    args = parser.parse_args()

    # If --no-features is specified, set show_features to False
    show_features = not args.no_features

    console = Console()

    # Create a table
    table = Table(title="LazySlide Model Parameter Sizes and Feature Dimensions")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Parameters", style="magenta", justify="right")

    # Add feature dimension column if requested
    if show_features:
        table.add_column("Feature Dim", style="yellow", justify="right")

    if args.detailed:
        table.add_column("License", style="yellow")
        table.add_column("Commercial", style="red")

    table.add_column("Module", style="blue")

    # Get models filtered by task if specified
    if args.task:
        model_keys = zs.models.list_models(task=args.task)
    else:
        model_keys = zs.models.list_models()

    # Further filter by model names if specified
    if args.models:
        # Convert to lowercase for case-insensitive matching
        model_names_lower = [name.lower() for name in args.models]
        # Filter model_keys to only include models whose names (case-insensitive) match any of the specified names
        model_keys = [
            key
            for key in model_keys
            if any(
                model_name.lower() in MODEL_REGISTRY[key].name.lower()
                for model_name in model_names_lower
            )
        ]

    # Track models we've already processed (to avoid duplicates)
    processed_models = set()

    for key in sorted(model_keys):
        card = MODEL_REGISTRY[key]

        # Skip if we've already processed this model
        if card.name in processed_models:
            continue
        processed_models.add(card.name)

        # Get model info from registry
        model_info = get_model_info_from_registry(card.name)

        # Get parameter count
        param_count = 0
        numeric_param_count = None
        feature_dim = None
        model_instance = None

        # Check if param_size is already in the registry
        if model_info and "param_size" in model_info and model_info["param_size"]:
            param_count = model_info["param_size"]
            console.print(
                f"[blue]Using parameter size from registry for {card.name}: {param_count}[/blue]"
            )
        elif not args.skip_load:
            try:
                # Try to import and instantiate the model
                model_class = card.module

                try:
                    model_instance = model_class()
                    numeric_param_count = count_parameters(model_instance)
                    param_count = human_readable_size(numeric_param_count)

                    # Update the registry if requested
                    if args.update_registry and numeric_param_count > 0:
                        if update_model_registry(card.name, param_size=param_count):
                            console.print(
                                f"[green]Updated registry for {card.name} with {param_count} parameters[/green]"
                            )
                        else:
                            console.print(
                                f"[red]Failed to update registry for {card.name}[/red]"
                            )
                except Exception as e:
                    console.print(
                        f"[red]Error loading model {card.name}: {str(e)}[/red]"
                    )
                    param_count = "Error"
            except Exception as e:
                console.print(f"[red]Error with model {card.name}: {str(e)}[/red]")
                param_count = "Error"
        else:
            param_count = "Not loaded"

        # Get feature dimension if requested and model is vision or multimodal
        if show_features and card.model_type[0].value in ["vision", "multimodal"]:
            # Check if encode_dim is already in the registry
            if model_info and "encode_dim" in model_info and model_info["encode_dim"]:
                feature_dim = model_info["encode_dim"]
                console.print(
                    f"[blue]Using feature dimension from registry for {card.name}: {feature_dim}[/blue]"
                )
            elif not args.skip_load and model_instance is not None:
                try:
                    # Determine feature dimension
                    if isinstance(model_instance, (ImageModel, ImageTextModel)):
                        dim = determine_feature_dimension(model_instance)
                        if dim is not None:
                            feature_dim = dim
                            console.print(
                                f"[green]Determined feature dimension for {card.name}: {feature_dim}[/green]"
                            )

                            # Update the registry if requested
                            if args.update_registry and feature_dim is not None:
                                if update_model_registry(
                                    card.name, feature_dim=feature_dim
                                ):
                                    console.print(
                                        f"[green]Updated registry for {card.name} with feature dimension {feature_dim}[/green]"
                                    )
                                else:
                                    console.print(
                                        f"[red]Failed to update registry for {card.name}[/red]"
                                    )
                        else:
                            feature_dim = "Unknown"
                    else:
                        feature_dim = "N/A"
                except Exception as e:
                    console.print(
                        f"[red]Error determining feature dimension for {card.name}: {str(e)}[/red]"
                    )
                    feature_dim = "Error"
            else:
                feature_dim = "Not determined"

        # Prepare row data
        row_data = [
            card.name,
            card.model_type[0].value,
            str(param_count),
        ]

        # Add feature dimension if requested
        if show_features:
            if card.model_type[0].value in ["vision", "multimodal"]:
                row_data.append(str(feature_dim) if feature_dim is not None else "N/A")
            else:
                row_data.append("N/A")

        # Add license and commercial info if detailed view is requested
        if args.detailed and model_info:
            license_info = model_info.get("license", "Unknown")
            if isinstance(license_info, list):
                license_info = ", ".join(license_info)

            commercial = model_info.get("commercial", False)
            commercial_str = "Yes" if commercial else "No"

            row_data.extend([license_info, commercial_str])
        elif args.detailed:
            row_data.extend(["Unknown", "Unknown"])

        # Add module name
        row_data.append(card.module.__name__)

        # Add row to table
        table.add_row(*row_data)

    # Print the table
    console.print(table)


if __name__ == "__main__":
    main()
