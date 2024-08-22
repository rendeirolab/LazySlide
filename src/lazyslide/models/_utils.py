def _fake_class(name, deps, inject=""):
    def init(self, *args, **kwargs):
        raise ImportError(
            f"To use {name}, you need to install {', '.join(deps)}."
            f"{inject}"
            "Please restart the kernel after installation."
        )

    # Dynamically create the class
    new_class = type(name, (object,), {"__init__": init})

    return new_class
