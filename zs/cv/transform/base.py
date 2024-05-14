# Base class
import numpy as np


class Transform:
    """Image Transform base class.

    Image -> Image
    """

    params: dict = {}

    def __repr__(self):
        # print params
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.__class__.__name__}({params_str})"

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
            processed_image = self.apply(image)
            return processed_image.astype(np.uint8)
        else:
            raise TypeError(f"Input must be np.ndarray, got {type(image)}")

    def apply(self, image):
        """Perform transformation"""
        raise NotImplementedError

    def set_params(self, **params):
        self.params.update(params)
