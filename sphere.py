import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objective_batch, measures_batch

from ribs.archives import GridArchive

max_bound = 100 / 2 * 5.12

archive = GridArchive(solution_dim=100,
                      dims=(100, 100),
                      ranges=[(-max_bound, max_bound), (-max_bound, max_bound)],
                      learning_rate=0.01,
                      threshold_min=0.0)

result_archive = GridArchive(solution_dim=100,
                             dims=(100, 100),
                             ranges=[(-max_bound, max_bound), (-max_bound, max_bound)])

from ribs.emitters import EvolutionStrategyEmitter

emitters = [
    EvolutionStrategyEmitter(
        archive,
        es="xnes",
        x0=np.zeros(100),
        sigma0=0.5,
        selection_rule="none",
        ranker="imp",
        batch_size=100, #has to be same as popsize
    ) for _ in range(5)
]

total_itrs = 15_000

from ribs.schedulers import Scheduler

scheduler = Scheduler(archive, emitters, result_archive=result_archive)


for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
    solution_batch = scheduler.ask()
    objective_batch, measure_batch = sphere(solution_batch)
    #print(f"SHAPE OF OBJECTIVE BATCH={objective_batch.shape}")
    #print(f"SHAPE OF MEASURE BATCH={measure_batch.shape}")
    scheduler.tell(objective_batch, measure_batch)

    # Output progress every 500 iterations or on the final iteration.
    if itr % 500 == 0 or itr == total_itrs:
        tqdm.write(f"Iteration {itr:5d} | "
                   f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}%  "
                   f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")
        
from ribs.visualize import grid_archive_heatmap

plt.figure(figsize=(8, 6))
grid_archive_heatmap(result_archive, vmin=0, vmax=100)

plt.show()