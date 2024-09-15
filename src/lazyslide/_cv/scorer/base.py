from collections import namedtuple

ScoreResult = namedtuple("ScoreResult", ["scores", "qc"])


class ScorerBase:
    """
    Base class for all scorers.

    All scores are operated on a patch.

    Image -> float
    """

    def __call__(self, patch, mask=None):
        return self.apply(patch, mask=None)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def apply(self, patch, mask=None) -> ScoreResult:
        """The scorer will return the scores and the bool value indicating of QC"""
        raise NotImplementedError


class ComposeScorer(ScorerBase):
    """
    Compose multiple scorers into one.

    Parameters
    ----------
    scorers : List[ScorerBase]
        List of scorers to be composed.
    """

    def __init__(self, scorers):
        self.scorers = scorers

    def apply(self, patch, mask=None) -> ScoreResult:
        scores = {}
        qc = True
        for scorer in self.scorers:
            score, _qc = scorer.apply(patch, mask)
            scores.update(score)
            qc &= _qc
        return ScoreResult(scores=scores, qc=qc)
