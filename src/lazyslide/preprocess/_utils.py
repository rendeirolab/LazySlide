from typing import Union

from lazyslide.cv.scorer import ScorerBase

Scorer = Union[ScorerBase, str]


def get_scorer(scorers):
    from lazyslide.cv.scorer import (
        Brightness,
        ComposeScorer,
        Contrast,
        FocusLite,
        Redness,
        ScorerBase,
    )

    scorer_mapper = {
        "focus": FocusLite,
        "contrast": Contrast,
        "brightness": Brightness,
        "redness": Redness,
    }

    scorer_list = []
    for s in scorers:
        if isinstance(s, ScorerBase):
            scorer_list.append(s)
        elif isinstance(s, str):
            scorer = scorer_mapper.get(s)
            if scorer is None:
                raise ValueError(
                    f"Unknown scorer {s}, "
                    f"available scorers are {'.'.join(scorer_mapper.keys())}"
                )
            # The scorer should be initialized when used
            scorer_list.append(scorer())
        else:
            raise TypeError(f"Unknown scorer type {type(s)}")
    compose_scorer = ComposeScorer(scorer_list)
    return compose_scorer
