"""Contains the DNSArchive."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from numpy_groupies import aggregate_nb as aggregate
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]
from scipy.spatial.distance import cdist

from ribs._utils import (
    check_batch_shape,
    check_finite,
    check_shape,
    validate_batch,
    validate_single,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._utils import fill_sentinel_values, parse_all_dtypes
from ribs.typing import BatchData, FieldDesc, Float, Int, SingleData


class DNSArchive(ArchiveBase):
    r"""An archive that maintains a fixed-size population via Dominated Novelty Search.

    Each generation, candidates are merged with the current population, and survivors
    are selected by their Dominated Novelty Search (DNS) score: for each solution, the
    DNS score is the mean distance in the chosen space to the k nearest neighbors with
    strictly higher objective ("fitter" neighbors). If no fitter neighbors exist, the
    DNS score is treated as ``+inf``.

    From the DNS paper: Bahlous-Boldi, R, and Faldor, M et Al. https://arxiv.org/abs/2502.00593

    By default, this archive stores the following data fields: ``solution``,
    ``objective``, ``measures``, and ``index``.

    Args:
        solution_dim: Dimensionality of the solution space. Scalar or multi-dimensional
            solution shapes are allowed by passing an empty tuple or tuple of integers,
            respectively.
        measure_dim: Dimensionality of the measure space.
        capacity: Fixed population size to maintain.
        k_neighbors: Number of fitter neighbors to average over when computing DNS.
        space: Space in which to compute distances ("measures" or "solution").
        qd_score_offset: Subtracted from objective values when computing QD score.
        seed: Value to seed the random number generator.
        solution_dtype: Data type of the solutions.
        objective_dtype: Data type of the objectives.
        measures_dtype: Data type of the measures.
        dtype: Shortcut to set all dtypes simultaneously.
        extra_fields: Extra fields to store alongside solutions.
        ckdtree_kwargs: Kwargs for :class:`scipy.spatial.cKDTree` used in retrieval.
    """

    def __init__(
        self,
        *,
        solution_dim: Int | tuple[Int, ...],
        measure_dim: Int,
        capacity: Int,
        k_neighbors: Int,
        qd_score_offset: Float = 0.0,
        seed: Int | None = None,
        solution_dtype: DTypeLike = None,
        objective_dtype: DTypeLike = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
        extra_fields: FieldDesc | None = None,
        ckdtree_kwargs: dict | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        # Set up the ArrayStore, which is a data structure that stores all the elites'
        # data in arrays sharing a common index.
        extra_fields = extra_fields or {}
        reserved_fields = {"solution", "objective", "measures", "index"}
        if reserved_fields & extra_fields.keys():
            raise ValueError(
                "The following names are not allowed in "
                f"extra_fields: {reserved_fields}"
            )
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        solution_dtype, objective_dtype, measures_dtype = parse_all_dtypes(
            dtype, solution_dtype, objective_dtype, measures_dtype, np
        )
        self._store = ArrayStore(
            field_desc={
                "solution": (self.solution_dim, solution_dtype),
                "objective": ((), objective_dtype),
                "measures": (self.measure_dim, measures_dtype),
                **extra_fields,
            },
            capacity=capacity,
        )

        # Set up constant properties.
        self._k_neighbors = int(k_neighbors)
        self._ckdtree_kwargs = {} if ckdtree_kwargs is None else ckdtree_kwargs.copy()
        self._qd_score_offset = np.asarray(
            qd_score_offset, dtype=self.dtypes["objective"]
        )

        # Set up k-D tree with current measures in the archive. Updated on add().
        self._cur_kd_tree = cKDTree(
            self._store.data("measures"), **self._ckdtree_kwargs
        )

        # Set up statistics -- objective_sum is the sum of all objective values in the
        # archive; it is useful for computing qd_score and obj_mean.
        self._best_elite = None
        self._objective_sum = None
        self._stats = None
        self._stats_reset()

    ## Properties inherited from ArchiveBase ##

    @property
    def field_list(self) -> list[str]:
        return self._store.field_list_with_index

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return self._store.dtypes_with_index

    @property
    def stats(self) -> ArchiveStats:
        return self._stats

    @property
    def empty(self) -> bool:
        return len(self._store) == 0

    ## Properties that are not in ArchiveBase ##
    ## Roughly ordered by the parameter list in the constructor. ##

    @property
    def best_elite(self) -> SingleData:
        """The elite with the highest objective in the archive.

        None if there are no elites in the archive.
        """
        return self._best_elite

    @property
    def k_neighbors(self) -> int:
        """The number of fitter neighbors for computing DNS."""
        return self._k_neighbors

    @property
    def space(self) -> str:
        """Space used for distance calculations ("measures" or "solution")."""
        return self._space

    @property
    def capacity(self) -> int:
        """Fixed number of solutions stored in this archive."""
        return self._store.capacity

    @property
    def cells(self) -> int:
        """Total capacity of the archive (for coverage/statistics)."""
        return self.capacity

    @property
    def qd_score_offset(self) -> float:
        """Subtracted from objective values when computing the QD score."""
        return self._qd_score_offset

    ## dunder methods ##

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[SingleData]:
        return iter(self._store)

    ## Utilities ##

    def _stats_reset(self) -> None:
        """Resets the archive stats."""
        self._best_elite = None
        self._objective_sum = np.asarray(0.0, dtype=self.dtypes["objective"])
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=np.asarray(0.0, dtype=self.dtypes["objective"]),
            qd_score=np.asarray(0.0, dtype=self.dtypes["objective"]),
            norm_qd_score=np.asarray(0.0, dtype=self.dtypes["objective"]),
            obj_max=None,
            obj_mean=None,
        )

    def _stats_update(self, new_objective_sum: Float, new_best_index: Float) -> None:
        """Updates statistics.

        Update is based on a new sum of objective values (new_objective_sum) and the
        index of a potential new best elite (new_best_index).
        """
        _, new_best_elite = self._store.retrieve([new_best_index])
        new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

        if (
            self._stats.obj_max is None
            or new_best_elite["objective"] > self._stats.obj_max
        ):
            self._best_elite = new_best_elite
            new_obj_max = new_best_elite["objective"]
        else:
            new_obj_max = self._stats.obj_max

        self._objective_sum = new_objective_sum
        new_qd_score = (
            self._objective_sum
            - np.asarray(len(self), dtype=self.dtypes["objective"])
            * self._qd_score_offset
        )
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=np.asarray(len(self) / self.cells, dtype=self.dtypes["objective"]),
            qd_score=new_qd_score,
            norm_qd_score=np.asarray(
                new_qd_score / self.cells, dtype=self.dtypes["objective"]
            ),
            obj_max=new_obj_max,
            obj_mean=np.asarray(
                self._objective_sum / len(self), dtype=self.dtypes["objective"]
            ),
        )

    def index_of(self, measures: ArrayLike) -> np.ndarray:
        """Returns the index of the closest solution to the given measures.

        Unlike the structured archives like :class:`~ribs.archives.GridArchive`, this
        archive does not have indexed cells where each measure "belongs." Thus, this
        method instead returns the index of the solution with the closest measure to
        each solution passed in.

        This means that :meth:`retrieve` will return the solution with the closest
        measure to each measure passed into that method.

        Args:
            measures: (batch_size, :attr:`measure_dim`) array of coordinates in measure
                space.

        Returns:
            (batch_size,) array of integer indices representing the location of the
            solution in the archive.

        Raises:
            RuntimeError: There were no entries in the archive.
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self.empty:
            raise RuntimeError(
                "There were no solutions in the archive. "
                "`DNSArchive.index_of` computes the nearest "
                "neighbor to the input measures, so there must be at least one "
                "solution present in the archive."
            )

        _, indices = self._cur_kd_tree.query(measures)
        return indices.astype(np.int32)

    def index_of_single(self, measures: ArrayLike) -> Int:
        """Returns the index of the measures for one solution.

        See :meth:`index_of`.

        Args:
            measures: (:attr:`measure_dim`,) array of measures for a single solution.

        Returns:
            Integer index of the measures in the archive's storage arrays.

        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    def compute_dns(self, measures: ArrayLike, objectives: ArrayLike) -> np.ndarray:
        """Computes DNS scores for a batch against the current population.

        For each input, finds distances to fitter neighbors (those in the archive with
        strictly higher objective) and returns the mean distance to the k nearest among
        them. If no fitter neighbors exist, the score is ``+inf``.
        """
        measures = np.asarray(measures)
        objectives = np.asarray(objectives)

        if self.empty:
            return np.full(len(measures), np.inf, dtype=self.dtypes["measures"])

        # Distances from batch to reference population.
        D = cdist(measures, self._store.data("measures"))

        ref_objectives = self._store.data("objective")
        # For each row i, mask columns where ref_objective < obj_i (not fitter or eq).
        scores = np.empty(len(measures), dtype=self.dtypes["measures"])
        for i, obj in enumerate(objectives):
            mask = ref_objectives >= obj
            if not np.any(mask):
                scores[i] = np.inf
                continue
            dists = D[i, mask]
            k = min(self._k_neighbors, dists.shape[0])
            # Take k smallest.
            part = np.partition(dists, k - 1)[:k]
            scores[i] = np.mean(part)
        return scores

    ## Methods for writing to the archive ##

    def _maybe_resize(self, new_size: int) -> None:
        """No-op for DNSArchive (fixed capacity)."""
        return

    def add(
        self,
        solution: ArrayLike,
        objective: ArrayLike | None,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> BatchData:
        """Inserts a batch of solutions with DNS-based survival selection.

        The current population and the incoming batch are merged, DNS scores are
        computed over the union, and the top ``capacity`` solutions by DNS are kept.
        """
        if objective is None:
            objective = np.zeros(len(solution), dtype=self.dtypes["objective"])

        data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        # Delete these so that we only use the clean, validated data in `data`.
        del solution, objective, measures, fields

        # Gather current population data.
        cur_size = len(self)
        if cur_size > 0:
            cur = self._store.data(return_type="dict")
        else:
            cur = {name: np.empty((0, *desc[0]), dtype=desc[1]) for name, desc in self._store.field_desc.items()}  # type: ignore[index]

        # Combine.
        combined = {}
        for name in self._store.field_list:
            combined[name] = (
                np.concatenate((cur[name], data[name]), axis=0)
                if name in data
                else cur[name]
            )

        # Compute DNS over union.
        X = combined["measures"]
        objs = combined["objective"]

        # Pairwise distances.
        if X.shape[0] == 0:
            dns_scores = np.array([], dtype=self.dtypes["measures"])  # pragma: no cover
        else:
            D = cdist(X, X)
            # Exclude self distances.
            np.fill_diagonal(D, np.inf)
            # For each i, consider only fitter neighbors (obj[j] >= obj[i]).
            fitter_mask = objs[None, :] >= objs[:, None]
            # Replace non-fitter distances with +inf.
            D_masked = np.where(fitter_mask, D, np.inf)
            # Compute mean of k smallest along each row (ignoring +inf when <k exist).
            dns_scores = np.empty(X.shape[0], dtype=self.dtypes["measures"])
            for i in range(X.shape[0]):
                row = D_masked[i]
                valid = np.isfinite(row)
                if not np.any(valid):
                    dns_scores[i] = np.inf
                    continue
                k = min(self._k_neighbors, np.sum(valid))
                part = np.partition(row[valid], k - 1)[:k]
                dns_scores[i] = np.mean(part)

        # Select survivors: top `capacity` by DNS (descending).
        cap = self.capacity
        n_total = dns_scores.shape[0]
        if n_total <= cap:
            survivor_indices = np.arange(n_total)
        else:
            # Take largest `cap` values.
            survivor_indices = np.argpartition(dns_scores, -cap)[-cap:]
            survivor_indices = survivor_indices[np.argsort(dns_scores[survivor_indices])]

        # Build add_info for batch entries.
        batch_size = len(data["measures"])
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "dns": np.empty(batch_size, dtype=self.dtypes["measures"]),
        }
        batch_start = cur_size
        batch_indices_in_union = np.arange(batch_start, batch_start + batch_size)
        batch_survivors = np.isin(batch_indices_in_union, survivor_indices)
        add_info["status"][batch_survivors] = 2
        add_info["dns"] = dns_scores[batch_indices_in_union]

        # Rebuild population as survivors (compact to [0..n-1]).
        survivors = {name: combined[name][survivor_indices] for name in self._store.field_list}
        self._store.clear()
        if survivors["measures"].shape[0] > 0:
            self._store.add(np.arange(survivors["measures"].shape[0]), survivors)

        # Update stats.
        if len(self) > 0:
            best_index = int(np.argmax(self._store.data("objective")))
            objective_sum = np.sum(self._store.data("objective"))
            self._stats_update(np.asarray(objective_sum, dtype=self.dtypes["objective"]), best_index)
            # Refresh KD-tree over measures.
            self._cur_kd_tree = cKDTree(self._store.data("measures"), **self._ckdtree_kwargs)

        return add_info

    def add_single(
        self,
        solution: ArrayLike,
        objective: ArrayLike | None,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> SingleData:
        """Inserts a single solution into the archive.

        Args:
            solution: Parameters of the solution.
            objective: Set to None to get the default value of 0; otherwise, a valid
                objective value is also acceptable.
            measures: Coordinates in measure space of the solution.
            fields: Additional data for the solution.

        Returns:
            Information describing the result of the add operation. The dict contains
            ``status`` and ``dns`` keys; refer to :meth:`add` for the meaning of
            status and dns.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures`` has
                non-finite values.
            ValueError: ``local_competition`` is turned on but objective was not passed
                in.
        """
        if objective is None:
            objective = 0.0

        data = validate_single(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        return self.add(**{key: [val] for key, val in data.items()})

    def clear(self) -> None:
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## Methods for reading from the archive ##
    ## Refer to ArchiveBase for documentation of these methods. ##

    def retrieve(self, measures: ArrayLike) -> tuple[np.ndarray, BatchData]:
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self._store.retrieve(self.index_of(measures))
        fill_sentinel_values(occupied, data)

        return occupied, data

    def retrieve_single(self, measures: ArrayLike) -> tuple[bool, SingleData]:
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])

        return occupied[0], {field: arr[0] for field, arr in data.items()}

    @overload
    def data(
        self,
        fields: str,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["dict"] = "dict",
    ) -> BatchData: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["tuple"] = "tuple",
    ) -> tuple[np.ndarray]: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["pandas"] = "pandas",
    ) -> ArchiveDataFrame: ...

    def data(
        self,
        fields: None | Collection[str] | str = None,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray | BatchData | tuple[np.ndarray] | ArchiveDataFrame:
        return self._store.data(fields, return_type)

    def sample_elites(self, n: Int) -> BatchData:
        if self.empty:
            raise IndexError("No elements in archive.")

        # Deterministic selection: return the first n elites (in storage order).
        # If n >= current population size, this returns the entire population.
        count = min(int(n), len(self._store))
        selected_indices = self._store.occupied_list[:count]
        _, elites = self._store.retrieve(selected_indices)
        return elites
