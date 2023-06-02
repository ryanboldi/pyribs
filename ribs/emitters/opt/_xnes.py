"""Implementation of Exponential NES: from Glasmachers, T., Schaul, T., Sun, Y., Wierstra, D., & Schmidhuber, J. (2010). Exponential natural evolution strategies. Annual Conference on Genetic and Evolutionary Computation.

Adapted from https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/xnes.py
"""
import numba as nb
import numpy as np
from threadpoolctl import threadpool_limits

from scipy.linalg import expm

from ribs._utils import readonly
from ribs.emitters.opt._evolution_strategy_base import EvolutionStrategyBase

class ExponentialNaturalEvolutionStrategy(EvolutionStrategyBase):
    """xNES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            covLearningRate=None,
            scaleLearningRate=None,
            centerLearningRate=None,
            dtype=np.float64):
        self.batch_size = (4 + int(3 * np.log(solution_dim))
                           if batch_size is None else batch_size)
        self.sigma0 = sigma0
        self.solution_dim = solution_dim
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)
        self._solutions = None

        # defaults for xNES
        self.covLearningRate = ((0.6)*(3+np.log(solution_dim))/(solution_dim*np.log(solution_dim))
                                if covLearningRate is None else covLearningRate)
        self.scaleLearningRate = ( self.covLearningRate
                                if scaleLearningRate is None else scaleLearningRate)
        self.centerLearningRate = (1
                                if centerLearningRate is None else centerLearningRate)

        # Strategy-specific params -> initialized in reset().
        self.current_eval = None
        self._A = None
        self._invA = None
        self._center = sigma0

    def reset(self, x0):
        """Resets the optimizer to start at x0.

        Args:
            x0 (np.ndarray): Initial mean.
        """
        self.current_eval = 0
        self.A = np.eye(self.solution_dim) #sqrt of the covariance matrix
        self._invA = np.eye(self.solution_dim) 
        self._center = x0

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to ``tell()``.

        Returns:
            True if any of the stopping conditions are satisfied.
        """
        if self.cov.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(np.max(self.cov.eigenvalues))
        if area < 1e-11:
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (len(ranking_values) >= 2 and
                np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    @staticmethod
    @nb.jit(nopython=True)
    def _transform_and_check_sol(sample, A, center, lower_bounds, upper_bounds):
        """Numba helper for transforming parameters to the solution space.

        Numba is important here since we may be resampling multiple times.
        """
        solutions = (np.dot(A, sample) + center)
        out_of_bounds = np.logical_or(
            solutions < np.expand_dims(lower_bounds, axis=0),
            solutions > np.expand_dims(upper_bounds, axis=0),
        )
        return solutions, out_of_bounds

    @staticmethod
    def _convert_base_to_sample(e, invA, center):
        "converting solutions from the solution space back to parameter space"
        return np.dot(invA, (e - center))

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
            batch_size (int): batch size of the sample. Defaults to
                ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        while len(remaining_indices) > 0:
            unscaled_params = self._rng.normal(
                0.0,
                1,
                (len(remaining_indices), self.solution_dim),
            ).astype(self.dtype)

            new_solutions, out_of_bounds = self._transform_and_check_sol(unscaled_params, 
            self._A, self._center, lower_bounds, upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]

        return readonly(self._solutions)

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def tell(self, ranking_indices, num_parents):
        """Passes the solutions back to the optimizer.

        Args:
            ranking_indices (array-like of int): Indices that indicate the
                ranking of the original solutions returned in ``ask()``.
            num_parents (int): Number of top solutions to select from the
                ranked solutions.
        """
        self.current_eval += len(self._solutions[ranking_indices])

        if num_parents == 0:
            return

        parents = self._convert_base_to_sample(self._solutions[ranking_indices][:num_parents])

        I = np.eye(self.solution_dim)

        utilities = [len(parents) - i for i in range(len(parents))] #simply monotonic ranking -> score; not sure about this
        dCenter = np.dot(parents.T, utilities)
        covGradient = np.dot((np.outer(parents, parents) - I).T, utilities)
        covTrace = np.trace(covGradient)
        covGradient -= covTrace / self.solution_dim * I
        dA = 0.5 * (self.scaleLearningRate * covTrace/self.solution_dim * I + self.covLearningRate * covGradient)
        
        self._center += self.centerLearningRate* np.dot(self._A, dCenter)
        self._A = np.dot(self._A, expm(dA))
        self._invA = np.dot(expm(-dA), self._invA)

        #self._logDetA += 0.5* self.scaleLearningRate * covTrace

