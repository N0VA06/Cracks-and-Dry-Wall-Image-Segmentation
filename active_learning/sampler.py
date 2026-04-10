"""
active_learning/sampler.py
Query strategy: select the K most uncertain unlabelled samples.
"""

import numpy as np
from torch.utils.data import Dataset, Subset


class ActiveLearningSampler:
    """
    Maintains the labelled / unlabelled index split and expands the
    labelled pool by querying the most uncertain samples each round.

    Args:
        dataset          : the full training dataset (all indices)
        initial_fraction : fraction used as labelled seed (round 0)
        query_fraction   : fraction of total dataset added per round
        seed             : numpy random seed
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_fraction: float = 0.10,
        query_fraction:   float = 0.10,
        seed: int = 42,
    ):
        self.dataset       = dataset
        self.query_fraction = query_fraction
        self.n_total        = len(dataset)

        rng = np.random.default_rng(seed)
        all_indices = rng.permutation(self.n_total).tolist()

        n_initial       = max(1, int(self.n_total * initial_fraction))
        self.labelled   = all_indices[:n_initial]
        self.unlabelled = all_indices[n_initial:]

        print(
            f"[ActiveLearningSampler] "
            f"Total: {self.n_total} | "
            f"Initial labelled: {len(self.labelled)} | "
            f"Unlabelled: {len(self.unlabelled)}"
        )

    # ------------------------------------------------------------------
    def get_labelled_subset(self) -> Subset:
        return Subset(self.dataset, self.labelled)

    def get_unlabelled_subset(self) -> Subset:
        return Subset(self.dataset, self.unlabelled)

    # ------------------------------------------------------------------
    def query(self, uncertainty_scores: np.ndarray, n_query: int | None = None) -> list[int]:
        """
        Select the top-K most uncertain unlabelled samples.

        Args:
            uncertainty_scores : scores for each sample in get_unlabelled_subset()
            n_query            : number to label; defaults to query_fraction * n_total
        Returns:
            list of original dataset indices that were selected
        """
        if n_query is None:
            n_query = max(1, int(self.n_total * self.query_fraction))
        n_query = min(n_query, len(self.unlabelled))

        # Top-K indices within the unlabelled scores array
        top_k_local = np.argsort(uncertainty_scores)[-n_query:][::-1]
        selected    = [self.unlabelled[i] for i in top_k_local]

        return selected

    # ------------------------------------------------------------------
    def expand_labelled(self, selected_indices: list[int]):
        """Move selected indices from unlabelled → labelled."""
        selected_set = set(selected_indices)
        self.labelled   = self.labelled + selected_indices
        self.unlabelled = [i for i in self.unlabelled if i not in selected_set]

        print(
            f"[Query] Added {len(selected_indices)} samples | "
            f"Labelled: {len(self.labelled)} | "
            f"Unlabelled: {len(self.unlabelled)}"
        )

    # ------------------------------------------------------------------
    @property
    def n_labelled(self) -> int:
        return len(self.labelled)

    @property
    def n_unlabelled(self) -> int:
        return len(self.unlabelled)
