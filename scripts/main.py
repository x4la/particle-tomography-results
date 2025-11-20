import plot
from scripts import benchmark

if __name__ == "__main__":

    benchmark.run_all_benchmarks() # run all methods on all datasets and save reconstructions in /out.
    # (Resire reconstructions should be precomputed and already reside in /out)

    # plot.plot_vesicle(show_3d_volumes=False)
    plot.plot_protein(show_3d_volumes=False)
    # plot.plot_thinfilm(show_3d_volumes=False)
    # plot.plot_platinum(show_3d_volumes=False)