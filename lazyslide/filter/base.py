class FilterBase:
    def get_scores(self, patch) -> float:
        raise NotImplementedError

    def filter(self, patch) -> bool:
        raise NotImplementedError
