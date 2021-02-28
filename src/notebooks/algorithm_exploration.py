# pylint: disable=invalid-name

import itertools
import timeit

import altair
import ipywidgets
import numpy
import pandas

from src.lib import divergence
from src.lib import normalize
from src.lib import randomness
from src.lib.approximation import dense
from src.lib.approximation.sparse import brute_force_search
from src.lib.approximation.sparse import compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import frank_wolfe
from src.lib.approximation.sparse import generalized_orthogonal_matching_pursuit
from src.lib.approximation.sparse import orthogonal_matching_pursuit
from src.lib.approximation.sparse import subspace_pursuit
from src.lib.approximation.sparse import warm_compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import warm_kl

# # Input

random_seed = 144

divergence_name = "total_variation"

selected_algorithms = {
    "CoSaMP, I=K, L=2K",
    "Frank-Wolfe, adaptive",
    "Frank-Wolfe, non-adaptive",
    "gOMP, L=2",
    "OMP",
    "SP, I=K, L=K",
    "Warm CoSaMP, I=K, L=2K",
    "Warm-KL, ηᵢ=1/(2i+1)",
    "Warm-KL, ηᵢ=1/(2K)",
}

density_range = (0, 1)
M = 16
N = 256

repetitions = 8

# # Calculation

generator = numpy.random.default_rng(random_seed)

D = getattr(divergence, divergence_name)

solve_dense = getattr(dense, divergence_name)

normalize_ = {
    "kullback_leibler": normalize.clip,
    "total_variation": normalize.clip,
}[divergence_name]

algorithms = {
    "Brute-force search": lambda *problem: brute_force_search.solve(
        *problem,
        solve_dense=solve_dense,
    ),
    "CoSaMP, I=K, L=2K": lambda A, b, D, K: compressive_sampling_matching_pursuit.solve(
        A,
        b,
        D,
        K,
        solve_dense=solve_dense,
        normalize=normalize_,
        I=K,
        L=min(2 * K, N),
    ),
    "Frank-Wolfe, adaptive": lambda *problem: frank_wolfe.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
        is_step_size_adaptive=True,
    ),
    "Frank-Wolfe, non-adaptive": lambda *problem: frank_wolfe.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
        is_step_size_adaptive=False,
    ),
    "gOMP, L=2": lambda *problem: generalized_orthogonal_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
        L=2,
    ),
    "OMP": lambda *problem: orthogonal_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
    ),
    "SP, I=K, L=K": lambda A, b, D, K: subspace_pursuit.solve(
        A,
        b,
        D,
        K,
        solve_dense=solve_dense,
        normalize=normalize_,
        I=K,
        L=K,
    ),
    "Warm CoSaMP, I=K, L=2K": lambda A, b, D, K: warm_compressive_sampling_matching_pursuit.solve(
        A,
        b,
        D,
        K,
        solve_dense=solve_dense,
        eta_i=lambda i: 1 / (2 * i + 1),
        normalize=normalize_,
        I=K,
        L=min(2 * K, N),
    ),
    "Warm-KL, ηᵢ=1/(2i+1)": lambda *problem: warm_kl.solve(
        *problem,
        solve_dense=solve_dense,
        eta_i=lambda i: 1 / (2 * i + 1),
    ),
    "Warm-KL, ηᵢ=1/(2K)": lambda A, b, D, K_: warm_kl.solve(
        A,
        b,
        D,
        K_,
        solve_dense=solve_dense,
        eta_i=lambda _: 1 / (2 * K_),
    ),
}

unknown_algorithms = selected_algorithms.difference(set(algorithms))
assert not unknown_algorithms, f"Unknown algorithms: {unknown_algorithms}"

data = pandas.DataFrame(columns=["K", "Algorithm", "Divergence", "Duration / s"])

progress = ipywidgets.FloatProgress(value=0.0, min=0.0, max=1.0)
progress  # pylint: disable=pointless-statement

for repetition in range(repetitions):
    progress.value = repetition / repetitions

    nonzero_range = (
        max(round(density_range[0] * M), 1),
        round(density_range[1] * M) + 1,
    )
    A = randomness.draw_distributions(generator, M, N, nonzero_range=nonzero_range)
    b = randomness.draw_distribution(generator, M)

    for K, algorithm in itertools.product(
        range(1, min(M, N)),
        selected_algorithms,
    ):
        solve = algorithms[algorithm]

        start_time = timeit.default_timer()
        x = solve(A, b, D, K)
        end_time = timeit.default_timer()

        assert x.shape == (N,)
        assert all(x >= 0)
        assert numpy.count_nonzero(x) <= K

        data = data.append(
            {
                "K": K,
                "Algorithm": algorithm,
                "Divergence": D(b, A @ x),
                "Duration / s": end_time - start_time,
            },
            ignore_index=True,
        )

progress.value = 1.0

# # Output

altair.Chart(data).mark_line().encode(
    x="K",
    y="mean(Divergence)",
    color="Algorithm",
)

altair.Chart(data).mark_line().encode(
    x="K",
    y="median(Divergence)",
    color="Algorithm",
)

altair.Chart(data).mark_boxplot().encode(
    x="Algorithm",
    y="Divergence",
    color="Algorithm",
    column="K",
)

altair.Chart(data).mark_line().encode(
    x="K",
    y="mean(Duration / s)",
    color="Algorithm",
)
