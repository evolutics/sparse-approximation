import altair
import ipywidgets
import numpy
import pandas

from src.lib import divergence
from src.lib import randomness


divergences = {
    "Euclidean": divergence.euclidean,
    "Hellinger": divergence.hellinger,
    "Jensen-Shannon distance": divergence.jensen_shannon_distance,
    "K directed": divergence.k_directed,
    "Kullback-Leibler": divergence.kullback_leibler,
    "Neyman χ²": divergence.neyman_chi_square,
    "Pearson χ²": divergence.pearson_chi_square,
    "Total variation": divergence.total_variation,
}

# # Divergences for Bernoulli distributions


def plot_bernoulli(p_0):
    p_1 = 1 - p_0
    p = numpy.array([p_0, p_1])

    q_0 = numpy.linspace(0, 1, num=100)
    Q = numpy.array([q_0, 1 - q_0])

    data = pandas.concat(
        [
            pandas.DataFrame({"Divergence": name, "q_0": Q[0, :], "D": D(p, Q)})
            for name, D in divergences.items()
        ]
    )
    data = data[data["D"] <= 1]

    return altair.Chart(data).mark_line().encode(x="q_0", y="D", color="Divergence")


ipywidgets.interact(plot_bernoulli, p_0=(0.0, 1.0, 0.01))

# # Example of disagreement between divergences


def get_example(d_0, d_1, size, tries, random_seed):
    generator = numpy.random.default_rng(random_seed)

    example = None
    example_differences = numpy.zeros(2)

    for _ in range(tries):
        p = randomness.draw_distribution(generator, size)
        q_0 = randomness.draw_distribution(generator, size)
        q_1 = randomness.draw_distribution(generator, size)

        d_0_0 = divergences[d_0](p, q_0)
        d_0_1 = divergences[d_0](p, q_1)
        d_1_0 = divergences[d_1](p, q_0)
        d_1_1 = divergences[d_1](p, q_1)

        differences = numpy.array([d_0_1 - d_0_0, d_1_0 - d_1_1])

        if all(differences > example_differences):
            example = pandas.DataFrame(
                {
                    "p": p,
                    f"{d_0} favors ({d_0_0:.3f} < {d_0_1:.3f})": q_0,
                    f"{d_1} favors ({d_1_0:.3f} > {d_1_1:.3f})": q_1,
                }
            )
            example_differences = differences

    return example


get_example(
    "Total variation", "K directed", size=3, tries=2**13, random_seed=144
).style.format("{:.3f}")
