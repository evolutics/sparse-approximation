import altair
import ipywidgets
import numpy
import pandas

from src.lib import divergence


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
