from os import makedirs
from os.path import exists
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)


def open_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def exist_create_folder(path):
    if not exists(path):
        makedirs(path)
    return 1


def res2mesh(results):
    res = []
    for i in results:
        if i != (-1, -1, -1):
            res.append(i)
    res = np.array(res)
    df = pd.DataFrame(res, columns=["ratio", "dec", "accuracy"])
    df = df.pivot(index="ratio", columns="dec", values="accuracy")
    return df


def plot_boxplot(gaussian, t, name):
    f, ax = plt.subplots(1, 1, figsize=(2 * 8.5 / 2.54, 2 * 4 / 2.54))
    sz = len(t)
    # t = [f"{i:.1e}" for i in t]
    data = np.where(gaussian[0] < 0, np.nan, gaussian[0])
    dataa = np.where(gaussian[1] < 0, np.nan, gaussian[1])

    temp = np.where(data == np.nan, np.nan, dataa)
    data = np.where(dataa == np.nan, np.nan, data)
    dataa = temp

    bp1 = ax.semilogx(
        t,
        np.nanmean(data, axis=1),
        color="black",
        marker="o",
        markersize=8,
        linestyle="",
        label="geometric",
    )
    bp2 = ax.semilogx(
        t,
        np.nanmean(dataa, axis=1),
        color="green",
        marker="d",
        markersize=8,
        linestyle="",
        label="arithmetic",
    )
    ax.grid(
        True,
        linestyle="--",
        linewidth=100 * 0.005,
        zorder=0,
        alpha=0.75,
        which="both",
        axis="both",
    )

    ax.set_xlabel("Level of noise ($\sigma^2$)", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylim(55, 100)

    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")
    return 1


def plot_mesh_perf(mesh, name):
    z = [res2mesh(mesh["arithmetic"]), res2mesh(mesh["geometric"])]
    names = ["Arithmetic", "Geometric"]
    levels = np.linspace(50, 100, 7)
    fig, ax = plt.subplots(
        1, 2, figsize=(3 * 8.5 / 2.54, 3 * 3.5 / 2.54), sharey=True, sharex=True
    )
    for i in range(2):
        x, y = np.meshgrid(z[i].columns.values, z[i].index.values)  # dec,ratio)
        m = ax[i].contourf(
            x, y, z[i].values, levels=levels, cmap="Blues_r", vmin=50, vmax=100
        )
        cntr = ax[i].contour(
            x,
            y,
            z[i].values,
            levels=levels,
            linewidths=0.5,
            colors=[str(i) for i in np.linspace(1, 0, 10)],
        )
        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].set_ylabel("ratio $R_{\\alpha}$", fontsize=14)
        ax[i].set_xlabel("distance $\Delta_m$", fontsize=14)
        ax[i].set_title(names[i], fontsize=14)
        ax[i].grid(color="red", linestyle="-.", alpha=0.3, linewidth=0.5)
        ax[i].set_ylim(1e-2, 1e2)
    fig.colorbar(
        m,
        ax=ax.ravel().tolist(),
        label="accuracy",
        location="bottom",
        pad=-0.45,
        shrink=0.8,
        aspect=30,
        extend="both",
    )
    plt.tight_layout()
    plt.savefig(f"{name}.pdf")
    return 1


if __name__ == "__main__":
    path_mesh = "../data/results_geom.pkl"
    path_addG = "../data/results_geom_noiseG.pkl"
    path_mulG = "../data/results_geom_noiseS.pkl"
    exist_create_folder("../figure/")

    try:
        mesh = open_pkl(path_mesh)
        plot_mesh_perf(mesh, "../figure/LR_mesh_comparison")
    except Exception as e:
        print(e)

    try:
        gaussian = np.array(open_pkl(path_addG))
        t = np.logspace(-4, np.log10(25), 20)
        plot_boxplot(gaussian, t, "../figure/add_noise_gaussian")
    except Exception as e:
        print(e)

    try:
        speckle = np.array(open_pkl(path_mulG))
        t = np.logspace(-4, np.log10(25), 20)
        plot_boxplot(speckle, t, "../figure/mul_noise_gaussian")
    except Exception as e:
        print(e)
