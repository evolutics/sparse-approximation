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

# # Input

random_seed = 144

divergence_name = "total_variation"

selected_algorithms = {
    "Brute-force search",
    "CoSaMP, I=K, L=2K",
    "Frank-Wolfe, adaptive",
    "Frank-Wolfe, non-adaptive",
    "gOMP, L=2",
    "OMP, clip",
    "OMP, id",
    "OMP, shift",
    "SP, I=K, L=K",
}

M = 16
N = 256

repetitions = 32

# # Calculation

generator = numpy.random.default_rng(random_seed)

D = getattr(divergence, divergence_name)

solve_dense = getattr(dense, divergence_name)

potential_clip = lambda r, A: D(normalize.clip(r), A)
potential_shift = lambda r, A: D(normalize.shift(r), A)

algorithms = {
    "Brute-force search": lambda *problem: brute_force_search.solve(
        *problem,
        D=D,
        solve_dense=solve_dense,
    ),
    "CoSaMP, I=K, L=2K": lambda A, b, K: compressive_sampling_matching_pursuit.solve(
        A,
        b,
        K,
        D=D,
        solve_dense=solve_dense,
        normalize=normalize.clip,
        I=K,
        L=min(2 * K, N),
    ),
    "Frank-Wolfe, adaptive": lambda *problem: frank_wolfe.solve(
        *problem,
        solve_dense=solve_dense,
        potential=potential_clip,
        is_step_size_adaptive=True,
    ),
    "Frank-Wolfe, non-adaptive": lambda *problem: frank_wolfe.solve(
        *problem,
        solve_dense=solve_dense,
        potential=potential_clip,
        is_step_size_adaptive=False,
    ),
    "gOMP, L=2": lambda *problem: generalized_orthogonal_matching_pursuit.solve(
        *problem,
        D=D,
        solve_dense=solve_dense,
        normalize=normalize.clip,
        L=2,
    ),
    "OMP, clip": lambda *problem: orthogonal_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        potential=potential_clip,
    ),
    "OMP, id": lambda *problem: orthogonal_matching_pursuit.solve(
        *problem, solve_dense=solve_dense, potential=D
    ),
    "OMP, shift": lambda *problem: orthogonal_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        potential=potential_shift,
    ),
    "SP, I=K, L=K": lambda A, b, K: subspace_pursuit.solve(
        A,
        b,
        K,
        D=D,
        solve_dense=solve_dense,
        normalize=normalize.clip,
        I=K,
        L=K,
    ),
}

unknown_algorithms = selected_algorithms.difference(set(algorithms))
assert not unknown_algorithms, f"Unknown algorithms: {unknown_algorithms}"

data = pandas.DataFrame(columns=["K", "Algorithm", "Divergence", "Duration / s"])

progress = ipywidgets.FloatProgress(value=0.0, min=0.0, max=1.0)
progress  # pylint: disable=pointless-statement

for repetition in range(repetitions):
    progress.value = repetition / repetitions

    A = randomness.draw_distributions(generator, M, N)
    b = randomness.draw_distribution(generator, M, nonzero_count=M)

    for K, algorithm in itertools.product(
        range(1, min(M, N)),
        selected_algorithms,
    ):
        solve = algorithms[algorithm]

        start_time = timeit.default_timer()
        x = solve(A, b, K)
        end_time = timeit.default_timer()

        assert x.shape == (N,)
        assert all((x >= 0) | numpy.isclose(x, 0))
        assert numpy.isclose(numpy.sum(x), 1)
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
