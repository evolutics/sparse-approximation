# pylint: disable=invalid-name, line-too-long

import itertools
import math
import timeit

import altair
import ipywidgets
import numpy
import pandas

from src.lib import divergence
from src.lib import normalize
from src.lib import randomness
from src.lib import sequence
from src.lib.approximation import dense
from src.lib.approximation.sparse import brute_force_search
from src.lib.approximation.sparse import compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import forward_backward_pursuit
from src.lib.approximation.sparse import frank_wolfe
from src.lib.approximation.sparse import generalized_orthogonal_matching_pursuit
from src.lib.approximation.sparse import generalized_reverse_matching_pursuit
from src.lib.approximation.sparse import multi_warm_js_subspace_pursuit
from src.lib.approximation.sparse import orthogonal_matching_pursuit
from src.lib.approximation.sparse import subspace_pursuit
from src.lib.approximation.sparse import warm_compressive_sampling_matching_pursuit
from src.lib.approximation.sparse import warm_js
from src.lib.approximation.sparse import warm_kl
from src.lib.approximation.sparse import warm_kl_compressive_sampling_matching_pursuit

altair.data_transformers.disable_max_rows()

# # Input

random_seed = 144

divergence_name = "total_variation"

selected_algorithms = {
    "CoSaMP, Lᵢ=2k, i∈[k]",
    "CoSaMP, Lᵢ=2k/2ⁱ",
    "FBP, α=2, β=1",
    "Frank-Wolfe, adaptive",
    "Frank-Wolfe, non-adaptive",
    "gOMP, L=2",
    "gRMP, exponential",
    "gRMP, linear",
    "Multi Warm-JS SP",
    "OMP",
    "SP, Lᵢ=k, i∈[k]",
    "SP, Lᵢ=k/2ⁱ",
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]",
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[k]",
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k/2ⁱ",
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]",
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[k]",
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k/2ⁱ",
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]",
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[k]",
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k/2ⁱ",
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]",
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[k]",
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k/2ⁱ",
    "Warm-JS, ηᵢ=1/(2i+1), I=2k",
    "Warm-JS, ηᵢ=1/(2k), I=2k",
    "Warm-JS, ηᵢ=D, I=2k",
    "Warm-KL, ηᵢ=1/(2i+1), I=4k",
    "Warm-KL, ηᵢ=1/(2k), I=4k",
    "Warm-KL, ηᵢ=D, I=4k",
}

density_range = (0, 1)
m = 16
n = 256

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
    "CoSaMP, Lᵢ=2k, i∈[k]": lambda C, p, D, k: compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        normalize=normalize_,
        L=[min(2 * k, n)] * k,
    ),
    "CoSaMP, Lᵢ=2k/2ⁱ": lambda C, p, D, k: compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        normalize=normalize_,
        L=sequence.halve_until_1(min(2 * k, n)),
    ),
    "FBP, α=2, β=1": lambda *problem: forward_backward_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
        alpha=2,
        beta=1,
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
    "gRMP, exponential": lambda *problem: generalized_reverse_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        L=lambda surplus: surplus // 2,
    ),
    "gRMP, linear": lambda C, p, D, k: generalized_reverse_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        L=round(n / k),
    ),
    "Multi Warm-JS SP": lambda C, p, D, k: multi_warm_js_subspace_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        etas=[1 / (2 * k), 1 / (8 * k), -2],
        I={2 * k, 8 * k},
        L=[k],
    ),
    "OMP": lambda *problem: orthogonal_matching_pursuit.solve(
        *problem,
        solve_dense=solve_dense,
        normalize=normalize_,
    ),
    "SP, Lᵢ=k, i∈[k]": lambda C, p, D, k: subspace_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        normalize=normalize_,
        L=[k] * k,
    ),
    "SP, Lᵢ=k/2ⁱ": lambda C, p, D, k: subspace_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        normalize=normalize_,
        L=sequence.halve_until_1(k),
    ),
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        normalize=normalize_,
        L=[min(2 * k, n)] * (math.floor(math.log2(k)) + 1),
    ),
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[k]": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        normalize=normalize_,
        L=[min(2 * k, n)] * k,
    ),
    "Warm CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k/2ⁱ": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        normalize=normalize_,
        L=sequence.halve_until_1(min(2 * k, n)),
    ),
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        normalize=normalize_,
        L=[min(2 * k, n)] * (math.floor(math.log2(k)) + 1),
    ),
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[k]": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        normalize=normalize_,
        L=[min(2 * k, n)] * k,
    ),
    "Warm CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k/2ⁱ": lambda C, p, D, k: warm_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        normalize=normalize_,
        L=sequence.halve_until_1(min(2 * k, n)),
    ),
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        L=[min(2 * k, n)] * (math.floor(math.log2(k)) + 1),
    ),
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k, i∈[k]": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        L=[min(2 * k, n)] * k,
    ),
    "Warm KL-CoSaMP, ηᵢ=1/(2i+1), I=4k, Lᵢ=2k/2ⁱ": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
        L=sequence.halve_until_1(min(2 * k, n)),
    ),
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[⌊log₂ k⌋+1]": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        L=[min(2 * k, n)] * (math.floor(math.log2(k)) + 1),
    ),
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k, i∈[k]": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        L=[min(2 * k, n)] * k,
    ),
    "Warm KL-CoSaMP, ηᵢ=D, I=4k, Lᵢ=2k/2ⁱ": lambda C, p, D, k: warm_kl_compressive_sampling_matching_pursuit.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
        L=sequence.halve_until_1(min(2 * k, n)),
    ),
    "Warm-JS, ηᵢ=1/(2i+1), I=2k": lambda C, p, D, k: warm_js.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=2 * k,
    ),
    "Warm-JS, ηᵢ=1/(2k), I=2k": lambda C, p, D, k: warm_js.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=1 / (2 * k),
        I=2 * k,
    ),
    "Warm-JS, ηᵢ=D, I=2k": lambda C, p, D, k: warm_js.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=2 * k,
    ),
    "Warm-KL, ηᵢ=1/(2i+1), I=4k": lambda C, p, D, k: warm_kl.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=-2,
        I=4 * k,
    ),
    "Warm-KL, ηᵢ=1/(2k), I=4k": lambda C, p, D, k: warm_kl.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=1 / (2 * k),
        I=4 * k,
    ),
    "Warm-KL, ηᵢ=D, I=4k": lambda C, p, D, k: warm_kl.solve(
        C,
        p,
        D,
        k,
        solve_dense=solve_dense,
        eta=None,
        I=4 * k,
    ),
}

unknown_algorithms = selected_algorithms.difference(set(algorithms))
assert not unknown_algorithms, f"Unknown algorithms: {unknown_algorithms}"

data = pandas.DataFrame(columns=["k", "Algorithm", "Divergence", "Duration / s"])

progress = ipywidgets.FloatProgress(value=0.0, min=0.0, max=1.0)
progress  # pylint: disable=pointless-statement

for repetition in range(repetitions):
    progress.value = repetition / repetitions

    nonzero_range = (
        max(round(density_range[0] * m), 1),
        round(density_range[1] * m) + 1,
    )
    C = randomness.draw_distributions(generator, m, n, nonzero_range=nonzero_range)
    p = randomness.draw_distribution(generator, m)

    for k, algorithm in itertools.product(
        range(1, min(m, n)),
        selected_algorithms,
    ):
        solve = algorithms[algorithm]

        start_time = timeit.default_timer()
        x = solve(C, p, D, k)
        end_time = timeit.default_timer()

        assert x.shape == (n,)
        assert all(x >= 0)
        assert numpy.count_nonzero(x) <= k

        data = pandas.concat(
            [
                data,
                pandas.DataFrame(
                    {
                        "k": [k],
                        "Algorithm": [algorithm],
                        "Divergence": [D(p, C @ x)],
                        "Duration / s": [end_time - start_time],
                    }
                ),
            ],
            ignore_index=True,
        )

progress.value = 1.0

# # Output

altair.Chart(data).mark_line().encode(
    x="k",
    y="mean(Divergence)",
    color="Algorithm",
)

altair.Chart(data).mark_line().encode(
    x="k",
    y="median(Divergence)",
    color="Algorithm",
)

altair.Chart(data).mark_boxplot().encode(
    x="Algorithm",
    y="Divergence",
    color="Algorithm",
    column="k",
).properties(height=1600)

altair.Chart(data).mark_line().encode(
    x="k",
    y="mean(Duration / s)",
    color="Algorithm",
)
