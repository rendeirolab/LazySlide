def _format_flops(flops: int) -> str:
    """Format FLOPS value in human-readable format."""
    if flops is None:
        return None
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    return str(flops)


# Model-specific input arguments for FLOPS estimation
MODEL_INPUT_ARGS_CONFIG = {
    "brightness": {"args": [(1, 3, 224, 224)]},
    "canny": {"args": [(1, 3, 224, 224)]},
    "cellpose": {"args": [(1, 3, 224, 224)]},
    "chief": {"args": [(1, 3, 224, 224)]},
    "chief-slide-encoder": {"args": [(100, 768)]},
    "conch": {"args": [(1, 3, 224, 224)], "method": "encode_image"},
    "conch_v1.5": {"args": [(1, 3, 448, 448)], "method": "conch.forward"},
    "contrast": {"args": [(1, 3, 224, 224)]},
    "ctranspath": {"args": [(1, 3, 224, 224)]},
    "entropy": {"args": [(1, 3, 224, 224)]},
    "focus": {"args": [(1, 3, 256, 256)]},
    "focuslitenn": {"args": [(1, 3, 256, 256)]},
    "gigapath": {"args": [(1, 3, 224, 224)]},
    "gigapath-slide-encoder": {"args": [(100, 1536), (100, 2)]},
    "gigatime": {"args": [(1, 3, 224, 224)]},
    "gpfm": {"args": [(1, 3, 224, 224)]},
    "grandqc-artifact": {"args": [(1, 3, 224, 224)]},
    "grandqc-tissue": {"args": [(1, 3, 224, 224)]},
    "h-optimus-0": {"args": [(1, 3, 224, 224)]},
    "h-optimus-1": {"args": [(1, 3, 224, 224)]},
    "h0-mini": {"args": [(1, 3, 224, 224)]},
    "haralick_texture": {"args": [(1, 3, 224, 224)]},
    "hest-tissue-segmentation": {"args": [(1, 3, 224, 224)]},
    "hibou-b": {"args": [], "kwargs": {"pixel_values": (1, 3, 224, 224)}},
    "hibou-l": {"args": [], "kwargs": {"pixel_values": (1, 3, 224, 224)}},
    "histoplus": {"args": [(1, 3, 840, 840)]},
    "instanseg": {"args": [(1, 3, 224, 224)]},
    "madeleine": {"args": [(1, 100, 512)]},
    "medsiglip": {"args": [], "kwargs": {"pixel_values": (1, 3, 448, 448)}, "method": "get_image_features"},
    "midnight": {"args": [(1, 3, 224, 224)]},
    "musk": {"args": [(1, 3, 384, 384)]},
    "nulite": {"args": [(1, 3, 224, 224)]},
    "omiclip": {"args": [(1, 3, 224, 224)]},
    "path_orchestra": {"args": [(1, 3, 224, 224)]},
    "pathprofiler": {"args": [(1, 3, 512, 512)]},
    "pathprofilerqc": {"args": [(1, 3, 224, 224)]},
    "phikon": {"args": [(1, 3, 224, 224)]},
    "phikonv2": {"args": [(1, 3, 224, 224)]},
    "plip": {"args": [], "kwargs": {"pixel_values": (1, 3, 224, 224)}, "method": "get_image_features"},
    "prism": {"args": [(1, 100, 768)], "method": "model.slide_representations"},
    "rosie": {"args": [(1, 3, 224, 224)]},
    "sam": {"args": [(1, 3, 1024, 1024)]},
    "saturation": {"args": [(1, 3, 224, 224)]},
    "sharpness": {"args": [(1, 3, 224, 224)]},
    "sobel": {"args": [(1, 3, 224, 224)]},
    "spider-breast": {"args": [(1, 3, 224, 224)]},
    "spider-colorectal": {"args": [(1, 3, 224, 224)]},
    "spider-skin": {"args": [(1, 3, 224, 224)]},
    "spider-thorax": {"args": [(1, 3, 224, 224)]},
    "split_rgb": {"args": [(1, 3, 224, 224)]},
    "titan": {"args": [(1, 3, 448, 448)], "method": "conch.forward"},
    "uni": {"args": [(1, 3, 224, 224)]},
    "uni2": {"args": [(1, 3, 224, 224)]},
    "virchow": {"args": [(1, 3, 224, 224)]},
    "virchow2": {"args": [(1, 3, 224, 224)]},
    "cytosyn": {"args": [], "method": "generate"},
}


def _get_default_flops_inputs(model) -> tuple[str, tuple, dict] | None:
    """Get default inputs for FLOPS estimation. Tries model-specific config first, then task-based defaults."""
    try:
        import torch
    except ImportError:
        return None
    from ._model_registry import MODEL_REGISTRY
    from .base import ModelTask

    model_class = model.__class__
    model_key = next((k for k, v in MODEL_REGISTRY.items() if v == model_class), None)
    if model_key and model_key in MODEL_INPUT_ARGS_CONFIG:
        config = MODEL_INPUT_ARGS_CONFIG[model_key]
        method = config.get("method", "forward")
        args = tuple(torch.randn(*shape) for shape in config.get("args", []))
        kwargs = {k: torch.randn(*v) for k, v in config.get("kwargs", {}).items()}
        return (method, args, kwargs)

    task = model.task[0] if isinstance(model.task, list) else model.task
    defaults = {
        ModelTask.vision: ("forward", (torch.randn(1, 3, 224, 224),), {}),
        ModelTask.segmentation: ("forward", (torch.randn(1, 3, 224, 224),), {}),
        ModelTask.multimodal: ("forward", (), {"pixel_values": torch.randn(1, 3, 224, 224)}),
        ModelTask.slide_encoder: ("forward", (torch.randn(100, 768),), {}),
        ModelTask.tile_prediction: ("forward", (torch.randn(1, 3, 224, 224),), {}),
        ModelTask.image_generation: ("generate", (), {}),
    }
    return defaults.get(task, ("forward", (torch.randn(1, 3, 224, 224),), {}))


def model_repr_html(model) -> str:
    """Return an HTML representation of the model card for Jupyter notebooks.

    This method is automatically called by Jupyter notebooks to display
    the model card as an HTML table.

    Returns
    -------
    str
        HTML representation of the model card.
    """
    # Create a styled HTML representation
    if isinstance(model.task, list):
        task = [m.value for m in model.task]
    else:
        task = [model.task.value]
    html = [
        '<div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; '
        'background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: fit-content;">',
        f'<h3 style="margin: 0 0 15px 0; text-align: center; color: #2c3e50;">{model.__class__.__name__}</h3>',
        '<ul style="list-style: none; padding: 0; margin: 0;">',
        '<li style="margin-bottom: 8px;">',
        f'<span style="font-weight: bold;">Model type:</span> {"; ".join(task)}</li>',
    ]

    # Add status with an icon
    status_icon = "🔒" if model.is_gated else "✅"
    status_text = "Gated" if model.is_gated else "Open"
    html.append(
        f'<li style="margin-bottom: 8px;"><span style="font-weight: bold;">'
        f"Status:</span> {status_icon} {status_text}</li>"
    )

    # Add FLOPS if available (static or computed)
    flops_value = None
    if hasattr(model, "flops") and model.flops is not None:
        # Use static value if available
        flops_value = model.flops
    elif hasattr(model, "estimate_flops") and hasattr(model, "model"):
        # Try to auto-compute FLOPS
        try:
            inputs = _get_default_flops_inputs(model)
            if inputs is not None:
                method, args, kwargs = inputs
                computed_flops = model.estimate_flops(method, *args, **kwargs)
                if computed_flops is not None:
                    flops_value = _format_flops(computed_flops)
        except Exception:
            # Silently fail if FLOPS can't be computed
            pass
    
    if flops_value is not None:
        html.append(
            f'<li style="margin-bottom: 8px;"><span style="font-weight: bold;">'
            f"FLOPS:</span> {flops_value}</li>"
        )

    # Add description if available
    if model.description:
        html.append(
            f'<li style="margin-bottom: 8px;"><span style="font-weight: bold;">'
            f"Description:</span> {model.description}</li>"
        )

    # Add a links section if any links are available
    links = []
    button_style = (
        "display: inline-block; padding: 6px 12px; margin: 0 6px 6px 0; border-radius: 4px; "
        "text-decoration: none; color: white; background-color: #3498db;"
    )

    if model.github_url:
        links.append(
            f'<a href="{model.github_url}" target="_blank" style="{button_style}">GitHub</a>'
        )
    if model.hf_url:
        links.append(
            f'<a href="{model.hf_url}" target="_blank" style="{button_style}">Hugging Face</a>'
        )
    if model.paper_url:
        links.append(
            f'<a href="{model.paper_url}" target="_blank" style="{button_style}">Paper</a>'
        )

    if links:
        html.append(f'<li style="margin-bottom: 8px;">{"".join(links)}</li>')

    # Close the HTML
    html.append("</ul></div>")

    return "".join(html)


def model_doc(model):
    skeleton = (
        ":octicon:`lock;1em;sd-text-danger;` "
        if model.is_gated
        else ":octicon:`check-circle-fill;1em;sd-text-success;` "
    )
    if model.hf_url is not None:
        skeleton += f":bdg-link-primary-line:`🤗Hugging Face <{model.hf_url}>` "
    if model.github_url is not None:
        skeleton += f":bdg-link-primary-line:`GitHub <{model.github_url}>` "
    if model.paper_url is not None:
        skeleton += f":bdg-link-primary-line:`Paper <{model.paper_url}>` "
    if model.param_size is not None:
        skeleton += f":bdg-info-line:`Params: {model.param_size}` "
    if model.flops is not None:
        skeleton += f":bdg-info-line:`FLOPS: {model.flops}` "
    if model.encode_dim:
        skeleton += f":bdg-info-line:`{model.encode_dim} features` "
    if model.license is not None:
        if isinstance(model.license, list):
            license_str = "; ".join(model.license)
        else:
            license_str = model.license
        if model.license_url is not None:
            skeleton += f":bdg-link-light:`{license_str} <{model.license_url}>` "
        else:
            skeleton += f":bdg-light:`{license_str}` "
    if model.bib_key is not None:
        skeleton += f":cite:p:`{model.bib_key}` "
    if model.description is not None:
        skeleton += f"\n{model.description}"

    return skeleton


def model_registry_repr_html(registry):
    """Return an HTML representation of the model registry for Jupyter notebooks."""
    if not registry._data:
        return f"<p>{type(registry).__qualname__}()</p>"

    # Get the DataFrame
    used_cols = [
        "is_gated",
        "key",
        "model_type",
        "github_url",
        "hf_url",
        "paper_url",
    ]
    display_cols = [
        "Gating Status",
        "Key",
        "Model Type",
        "GitHub",
        "HuggingFace",
        "Paper",
    ]
    df = registry.to_dataframe().loc[:, used_cols]

    def format_is_gated(value):
        # Convert boolean is_gated to emoji icon
        return "🔒" if value else "✅"

    # Define a function to style rows based on is_gated value
    def style_rows(row):
        bg_color = (
            "pink" if row["is_gated"] else "#e8f5e9"
        )  # Red for gated, green for open
        return [f"background-color: {bg_color}"] * len(row)

    # Define a function to format URL columns as link buttons
    def format_url(value):
        if value is None:
            return ""
        return f'<a href="{value}" target="_blank">Link</a>'

    # Create a styler object
    styler = (
        df.style
        # Apply row styling based on is_gated
        .apply(style_rows, axis=1)
        # Format URL columns as link buttons
        .format(
            {
                "hf_url": format_url,
                "paper_url": format_url,
                "github_url": format_url,
                "is_gated": format_is_gated,
            },
            escape="html",
        )
        # Rename the column name
        .relabel_index(display_cols, axis="columns")
        # Hide the index
        .hide(axis="index")
    )

    return styler.to_html()
