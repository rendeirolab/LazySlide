import numpy as np


class ScorerBase:
    """
    Base class for all scorers.

    All scores are operated on a patch.

    Image -> float
    """

    threshold: float
    name: str = "base_score"

    def __call__(self, patch):
        return self.get_score(patch)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def get_score(self, patch) -> float:
        """Get scores for a patch"""
        raise NotImplementedError

    def get_scores(self, patches) -> np.ndarray:
        """Get scores for a list of patches"""
        scores = [self.get_score(patch) for patch in patches]
        return np.array(scores)
