from __future__ import annotations

from typing import Any, Literal


class RNALinker:
    """
    Link the aggregated WSI features with other omics data.

    Parameters
    ----------
    agg_features : AnnData
        The aggregated WSI features
    others : AnnData
        Other omics data, like RNA-seq, must have the same number of observations as agg_features.
    gene_name : str, default: None
        The key to use for the name of the omics data.

    """

    def __init__(
        self,
        agg_features: Any,
        others: Any,
        gene_name: str | None = None,
    ):
        try:
            import scanpy as sc
            from anndata import AnnData
        except ImportError:
            raise ImportError(
                "To use MultimodalLinker, you need to install scanpy. You can install it using "
                "`pip install scanpy."
            )

        if not isinstance(agg_features, AnnData) or not isinstance(others, AnnData):
            raise TypeError("agg_features and others must be AnnData objects")

        self.agg_features = agg_features
        self.others = others
        self.gene_name = gene_name
        self.groupby = None
        self.score_group = None
        self.score_key = None
        self.association_score_key = "association_score"

    def score(
        self,
        groupby: str,
        score_group: str,
        scale: bool = True,
        test_method: str = "t-test",
        n_features: int | str = 100,
        key_added: str | None = None,
    ):
        """
        Score a group of samples based on the aggregated features.

        The score will be higher for the score group and lower for the rest.

        Parameters
        ----------
        groupby : str
            The key in the obs of agg_features.
        score_group : str
            The specific group in the column of groupby to score.
        scale : bool, default: True
            Scale the features between -1 and 1.
        test_method : str, default: "t-test"
            The method to use for ranking omics features like genes.
        n_features : int | str, default: 100
            The number of features to use for scoring. If "all", use all features.
        key_added : str | None, default: None
            The key to store the scores in the obs of agg_features.
            If not specify, f"{score_group}_score" will be added to the obs of agg_features.

        """
        import numpy as np
        import scanpy as sc

        self.groupby = groupby
        self.score_group = score_group
        self.score_key = key_added or f"{score_group}_score"

        sc.tl.rank_genes_groups(self.agg_features, groupby=groupby, method=test_method)

        group_df = sc.get.rank_genes_groups_df(self.agg_features, group=score_group)
        # Get the top n_features and low n_features
        if n_features == "all":
            features = self.agg_features.X
            weights = group_df["scores"]
        else:
            top = group_df.head(n_features)
            low = group_df.tail(n_features)
            top_names = top["names"].astype(int)
            low_names = low["names"].astype(int)

            sel = list(top_names) + list(low_names)
            features = self.agg_features.X[:, sel]

            if scale:
                # scale each feature between -1 and 1
                features = (features - features.min(0)) / (
                    features.max(0) - features.min(0)
                )
                features = 2 * features - 1

            weights = top["scores"].to_list() + low["scores"].to_list()
            weights = np.array(weights)

        # Score cells
        scores = features @ weights
        self.agg_features.obs[self.score_key] = scores

    def plot_score(self, ax=None):
        """
        Plot the score distribution for the score group and others.
        """
        import pandas as pd
        import seaborn as sns
        from matplotlib import pyplot as plt

        if self.score_key is None:
            raise ValueError("Please run .score() first.")

        if ax is None:
            _, ax = plt.subplots(figsize=(2, 4))

        groups = self.agg_features.obs[self.groupby].to_numpy()
        groups[groups != self.score_group] = "others"

        df = pd.DataFrame(
            {
                "score": self.agg_features.obs[self.score_key],
                "group": groups,
            }
        )
        sns.boxplot(
            data=df, x="group", y="score", ax=ax, color="#C68FE6", showfliers=False
        )
        sns.stripplot(data=df, x="group", y="score", ax=ax, color="#C68FE6", alpha=0.5)
        sns.despine()

    def associate(
        self,
        method: Literal[
            "pearson", "spearman", "kendall", "linear_reg", "lasso"
        ] = "linear_reg",
        score_key: str = None,
        key_added: str = "association_score",
    ):
        """
        Associate scores with other omics.
        """
        import numpy as np
        import pandas as pd
        from anndata import AnnData

        omics_matrix = self.others
        if not isinstance(omics_matrix, AnnData):
            raise TypeError("omics_matrix must be an AnnData object.")
        # Check if the shape matches
        if omics_matrix.n_obs != self.agg_features.n_obs:
            raise ValueError(
                "The number of observations in omics_matrix "
                "must match the number of observations in "
                "agg_features."
            )

        omics_df = pd.DataFrame(omics_matrix.X)
        scores = pd.Series(
            self.agg_features.obs[score_key].to_numpy(), index=omics_df.index
        )

        if method in {"pearson", "spearman", "kendall"}:
            coef = omics_df.corrwith(scores, method=method)
        elif method == "linear_reg":
            from sklearn.linear_model import LinearRegression

            clf = LinearRegression()
            clf.fit(omics_df, scores)
            coef = clf.coef_
        elif method == "lasso":
            from sklearn.linear_model import Lasso

            clf = Lasso()
            clf.fit(omics_df, scores)
            coef = clf.coef_
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

        if key_added is None:
            key_added = self.association_score_key
        else:
            self.association_score_key = key_added

        omics_matrix.var[key_added] = np.asarray(coef)

    def _get_associated_genes(self, gene_name: str | None = None):
        # Import dependencies locally
        import pandas as pd

        omics_matrix = self.others
        if self.association_score_key not in omics_matrix.var:
            raise ValueError("Please run .associate() first.")

        if gene_name is not None:
            index = omics_matrix.var[gene_name]
        else:
            index = omics_matrix.var.index

        scores_df = pd.DataFrame(
            {
                self.association_score_key: omics_matrix.var[
                    self.association_score_key
                ].values
            },
            index=index,
        )
        return scores_df.sort_values(self.association_score_key, ascending=False)

    def plot_rank(
        self,
        n_genes=5,
        gene_name: str | None = None,
    ):
        """
        Plot the rank of the association score.
        """
        import numpy as np
        from matplotlib import pyplot as plt

        omics_matrix = self.others
        if self.association_score_key not in omics_matrix.var:
            raise ValueError("Please run .associate() first.")

        # Plot a rank plot where x is the association score and y is the rank
        # sort the data
        scores_df = self._get_associated_genes(gene_name)
        scores_df["rank"] = np.arange(len(scores_df))

        _, ax = plt.subplots()
        ax.plot(
            scores_df["rank"], scores_df[self.association_score_key], color="#C68FE6"
        )
        ax.axhline(0, color="black", linestyle="--")

        # show the top n_var
        texts = "\n".join(scores_df.index[:n_genes])
        row = scores_df.iloc[0]
        ax.annotate(
            texts,
            (row["rank"], row[self.association_score_key]),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="top",
        )

        # show the bottom n_var
        texts = "\n".join(scores_df.index[-n_genes:])
        row = scores_df.iloc[-1]
        ax.annotate(
            texts,
            (row["rank"], row[self.association_score_key]),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="right",
            va="bottom",
        )
        ax.set(
            xlabel="Rank",
            ylabel="WSI Association Score",
        )

    def associated_genes(
        self,
        n_genes=5,
        gene_name: str | None = None,
    ):
        """
        Get the top and bottom associated genes.
        """
        scores_df = self._get_associated_genes(gene_name)
        return {"top": scores_df.head(n_genes), "bottom": scores_df.tail(n_genes)}
