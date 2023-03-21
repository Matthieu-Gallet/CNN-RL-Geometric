from weibull_sensitivity import *

if __name__ == "__main__":
    output = "../data/"
    makedirs(output, exist_ok=True)

    seed = 42
    np.random.seed(seed)

    hypercube_L = sample_lhs(33, lower=1e-2, upper=1e2)
    ratio, shift = hypercube_L[:, 0], hypercube_L[:, 1]
    print(hypercube_L.shape)

    noises_r = np.logspace(-4, np.log10(25), 20)
    for t in ["speckle", "gaussian"]:
        resg, resa = [], []
        for noise in noises_r:
            geometric = Parallel(n_jobs=-1)(
                delayed(sample_weibull)(r, d, "geometric", noise, t)
                for r in ratio
                for d in shift
            )
            arithmetic = Parallel(n_jobs=-1)(
                delayed(sample_weibull)(r, d, "arithmetic", noise, t)
                for r in ratio
                for d in shift
            )
            resg.append(np.array(geometric)[:, -1])
            resa.append(np.array(arithmetic)[:, -1])
        dump_pkl([resg, resa], join(output, f"weibull_noise_{t}.pkl"))
