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
