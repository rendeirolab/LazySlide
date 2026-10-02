from typing import TYPE_CHECKING, Literal

from ._utils import get_torch_device, warn_deprecated

if TYPE_CHECKING:
    import torch


class Settings:
    # Still accepted by ``settings[...]`` until removal, but no longer listed.
    _deprecated = frozenset({"pbar_impl"})

    def __init__(self) -> None:
        # Route initial values through setters to ensure validation on construction.
        # ``autocast_dtype`` and ``device`` are intentionally NOT set here so that
        # ``import lazyslide`` does not import torch; they resolve lazily on first
        # access (see their property getters).
        self.amp = False
        self.compile = False
        self.compile_kws = None
        self.pbar = True

    @property
    def _attributes(self) -> set[str]:
        return {
            "amp",
            "autocast_dtype",
            "compile",
            "compile_kws",
            "device",
            "pbar",
        }

    # amp
    @property
    def amp(self) -> bool:  # type: ignore[override]
        return getattr(self, "_amp", False)

    @amp.setter
    def amp(self, value) -> None:  # type: ignore[override]
        if isinstance(value, bool):
            self._amp = value
            return
        if value in (0, 1):  # allow 0/1 as booleans
            self._amp = bool(value)
            return
        raise TypeError("amp must be a boolean.")

    # compile
    @property
    def compile(self) -> bool:  # type: ignore[override]
        return getattr(self, "_compile", False)

    @compile.setter
    def compile(self, value) -> None:  # type: ignore[override]
        if isinstance(value, bool):
            self._compile = value
            return
        if value in (0, 1):  # allow 0/1 as booleans
            self._compile = bool(value)
            return
        raise TypeError("compile must be a boolean.")

    # compile_kws
    @property
    def compile_kws(self) -> dict:  # type: ignore[override]
        # Copy on read, mirroring the copy on write in the setter: callers must
        # not be able to mutate the stored settings through the returned dict.
        return dict(getattr(self, "_compile_kws", None) or {})

    @compile_kws.setter
    def compile_kws(self, value) -> None:  # type: ignore[override]
        if value is None:
            self._compile_kws = {}
            return
        if not isinstance(value, dict):
            raise TypeError("compile_kws must be a dict or None.")
        self._compile_kws = dict(value)

    # autocast_dtype
    @property
    def autocast_dtype(self) -> "torch.dtype":  # type: ignore[override]
        val = getattr(self, "_autocast_dtype", None)
        if val is None:
            import torch

            val = torch.float16
        return val

    @autocast_dtype.setter
    def autocast_dtype(self, value) -> None:  # type: ignore[override]
        import torch

        allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        if not isinstance(value, torch.dtype):
            if isinstance(value, str):
                normalized = value.lower().replace("fp", "float").strip()
                mapping = {
                    "float16": torch.float16,
                    "half": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    "single": torch.float32,
                }
                if normalized in mapping:
                    value = mapping[normalized]
                else:
                    raise TypeError(
                        "autocast_dtype must be a torch.dtype or one of: 'float16', 'bfloat16', 'float32'."
                    )
            else:
                raise TypeError(
                    "autocast_dtype must be a torch.dtype or a supported string."
                )
        if value not in allowed_dtypes:
            allowed = ", ".join(str(d) for d in allowed_dtypes)
            raise ValueError(f"autocast_dtype must be one of: {allowed}.")
        self._autocast_dtype = value

    @property
    def device(self) -> "torch.device":
        val = getattr(self, "_device", None)
        if val is None:
            val = get_torch_device()
            self._device = val
        return val

    @device.setter
    def device(self, value) -> None:
        import torch

        self._device = torch.device(value)

    # pbar
    @property
    def pbar(self) -> bool:  # type: ignore[override]
        return getattr(self, "_pbar", True)

    @pbar.setter
    def pbar(self, value) -> None:  # type: ignore[override]
        if isinstance(value, bool):
            self._pbar = value
            return
        if value in (0, 1):
            self._pbar = bool(value)
            return
        raise TypeError("pbar must be a boolean.")

    # pbar_impl
    @property
    def pbar_impl(self) -> Literal["tqdm", "rich"]:  # type: ignore[override]
        return getattr(self, "_pbar_impl", "rich")

    @pbar_impl.setter
    def pbar_impl(self, value) -> None:  # type: ignore[override]
        warn_deprecated(
            "`settings.pbar_impl` is deprecated since v0.13.0 and will be removed in "
            "v0.14.0; it has no effect, progress bars always use rich."
        )
        if not isinstance(value, str):
            raise TypeError("pbar_impl must be a string 'tqdm' or 'rich'.")
        value = value.strip().lower()
        if value not in ("tqdm", "rich"):
            raise ValueError("pbar_impl must be either 'tqdm' or 'rich'.")
        self._pbar_impl = value  # type: ignore[assignment]

    def __getitem__(self, key):
        if key not in self._attributes | self._deprecated:
            raise KeyError(f"{key} is not a valid setting.")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self._attributes | self._deprecated:
            raise KeyError(f"{key} is not a valid setting.")
        setattr(self, key, value)

    def __repr__(self):
        title = "LazySlide Settings"
        contents = "\n".join(
            [f"{attr}: {getattr(self, attr)}" for attr in self._attributes]
        )
        return f"{title}\n==================\n{contents}"

    # HTML representation for rich display in notebooks (Jupyter/IPython)
    def _repr_html_(self) -> str:
        """Return a compact HTML table describing the current settings.

        This is used by Jupyter frontends for rich display. It intentionally has
        no external dependencies and keeps styling inline.
        """
        # Local import to avoid polluting module namespace when not used
        import html as _html

        import torch

        def _fmt(value) -> str:
            # Pretty booleans with icons
            if isinstance(value, bool):
                return f"{'✅' if value else '❌'} {value}"
            # Render torch dtypes as code with the short name (e.g., float16)
            if isinstance(value, torch.dtype):
                short = str(value).split(".")[-1]
                return f"<code>{_html.escape(short)}</code>"
            # Fallback: plain-escaped string
            return _html.escape(str(value))

        rows = []
        for key in self._attributes:
            val = getattr(self, key)
            rows.append(f"<tr><th>{key}</th><td>{_fmt(val)}</td></tr>")

        # Simple inline styles to look nice in light/dark themes
        base_style = (
            "font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; "
            "font-size: 13px; line-height: 1.4;"
        )
        table_style = (
            "border-collapse: collapse; border: 1px solid var(--jp-border-color2, #ddd); "
            "background: var(--jp-layout-color1, #fff);"
        )
        th_style = (
            "text-align: left; padding: 6px 8px; white-space: nowrap; "
            "background: var(--jp-layout-color2, #f7f7f9); "
            "border-bottom: 1px solid var(--jp-border-color2, #eee);"
        )
        td_style = (
            "padding: 6px 8px; border-bottom: 1px solid var(--jp-border-color2, #eee);"
        )
        caption = (
            "<caption style='caption-side: top; text-align:left; font-weight:600; padding:4px 0;'>"
            "LazySlide Settings"
            "</caption>"
        )

        # Inject styles into tags
        body = (
            "".join(rows)
            .replace("<th>", f"<th style='{th_style}'>")
            .replace("<td>", f"<td style='{td_style}'>")
        )

        return (
            f"<div style='{base_style}'>"
            f"<table style='{table_style}'>"
            f"{caption}"
            f"{body}"
            f"</table>"
            f"</div>"
        )


settings = Settings()
