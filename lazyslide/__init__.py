import lazy_loader as lazy

__version__ = "0.1.0"

subpackages = ["pp", "tl", "pl", "get", "models"]


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=subpackages,
    submod_attrs={
        "wsi": ["WSI"],
    },
)

# from .wsi import WSI
# import lazyslide.pp as pp
# import lazyslide.tl as tl
# import lazyslide.pl as pl
# import lazyslide.get as get
