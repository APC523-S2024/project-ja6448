# APC 523 - Project - Josh Arrington - Spring 2024
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from cycler import cycler
import time

from allen_cahn_1d import AllenCahn1D
from allen_cahn_2d import AllenCahn2D

plt.rcParams.update({"font.size": 12})


def plot_errors(times, errors, Ns, usr_title=0, saveflag=0):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(Ns)):
        N = Ns[i]
        t = times[i][1:]
        err = errors[i]
        ax.plot(t, err, label="N = {}".format(N))

    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"L2 error, $e$")
    fig.legend(loc="lower left", bbox_to_anchor=[0.73, 0.11])
    ax.set_yscale("log")
    ax.set_xscale("log")
    if usr_title != 0:
        ax.set_title(usr_title)
    fig.tight_layout()

    if saveflag != 0:
        savestr = usr_title.split(" ")
        savestr = "_".join(savestr)
        plt.savefig("../figures/" + savestr + ".png")

    return fig, ax


def compare_errors(
    times, errors, Ns, usr_title=0, saveflag=0, usr_labels=["FE", "fft"]
):
    line_opts = ["-", "--", ":", "-."]
    color_table = mcolor.TABLEAU_COLORS
    ctable = []
    for c in color_table:
        ctable.append(c)
    Nint = int(len(times) / len(Ns))
    t_new = []
    e_new = []
    for i in range(len(Ns)):
        for j in range(Nint):
            t_new.append(times[i + j * len(Ns)])
            e_new.append(errors[i + j * len(Ns)])

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(Ns)):
        N = Ns[i]
        for j in range(Nint):
            t = times[i + j * len(Ns)][1:]
            err = errors[i + j * len(Ns)]
            leg_label = "{}: N = {}".format(usr_labels[j], N)
            ax.plot(t, err, color=ctable[i], label=leg_label, linestyle=line_opts[j])

    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"L2 error, $e$")
    fig.legend(loc="lower left", bbox_to_anchor=[0.334, 0.11], ncols=5, fontsize=10)
    ax.set_yscale("log")
    ax.set_xscale("log")
    if usr_title != 0:
        ax.set_title(usr_title)
    fig.tight_layout()

    if saveflag != 0:
        savestr = usr_title.split(" ")
        savestr = "_".join(savestr)
        plt.savefig("../figures/" + savestr + ".png")

    return fig, ax


def analysis_1D():
    # 1D params
    Nx = [32, 64, 128, 256, 512]
    L = 50
    w = 1.0
    H = 1.0
    tau = 1.0
    end_t = 20.0
    zero_loc = 10.0
    fe_times = []
    fe_errors = []
    fe_exec_times = []
    fft_times = []
    fft_errors = []
    fft_exec_times = []
    RK4_times = []
    RK4_errors = []
    RK4_exec_times = []

    for i, N in enumerate(Nx):
        print(10 * "-" + "1D, N = {} ".format(N) + 10 * "-")
        dt = (0.005 / (50 / 64)) * (L / N)
        AC1D = AllenCahn1D(L, zero_loc, N, w, H, tau)

        # Forward Euler
        fe_time = time.time()
        if N == 32:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.FE_step, dt, end_t, gen_prof=1, prof_title="FE"
            )

        else:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.FE_step, dt, end_t
            )
        print("done with FE, exec time = {:0.3f}".format(time.time() - fe_time))
        fe_times.append(tmp_times), fe_errors.append(tmp_errors), fe_exec_times.append(
            tmp_exec_times
        )

        # Reinitialize
        AC1D.__init__(L, zero_loc, N, w, H, tau)

        # FFT
        AC1D.fft_init()
        fft_time = time.time()
        if N == 32:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.fft_step, dt, end_t, gen_prof=1, prof_title="FFT"
            )
        else:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.fft_step, dt, end_t
            )
        print("done with fft, exec time = {:0.3f}".format(time.time() - fft_time))
        fft_times.append(tmp_times), fft_errors.append(
            tmp_errors
        ), fft_exec_times.append(tmp_exec_times)

        # Reinitialize
        AC1D.__init__(L, zero_loc, N, w, H, tau)

        # RK4
        RK4_time = time.time()
        if N == 32:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.RK4_step, dt, end_t, gen_prof=1, prof_title="RK4"
            )
        else:
            tmp_times, tmp_errors, tmp_exec_times = AC1D.integrate(
                AC1D.RK4_step, dt, end_t
            )
        print("done with RK4, exec time = {:0.3f}".format(time.time() - RK4_time))
        RK4_times.append(tmp_times), RK4_errors.append(
            tmp_errors
        ), RK4_exec_times.append(tmp_exec_times)

    plot_errors(fe_times, fe_errors, Nx, usr_title="FE errors 1D", saveflag=1)
    plot_errors(fft_times, fft_errors, Nx, usr_title="FFT errors 1D", saveflag=1)
    plot_errors(RK4_times, RK4_errors, Nx, usr_title="RK4 errors 1D", saveflag=1)

    times = fe_times
    errors = fe_errors
    for i in range(len(Nx)):
        times.append(fft_times[i])
        errors.append(fft_errors[i])

    for i in range(len(Nx)):
        times.append(RK4_times[i])
        errors.append(RK4_errors[i])
    compare_errors(
        times,
        errors,
        Nx,
        usr_labels=["FE", "FFT", "RK4"],
        saveflag=1,
        usr_title="AC1D error comparison",
    )

    # execution times
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(Nx, fe_exec_times, "g*", label="FE")
    ax.plot(Nx, fft_exec_times, "r^", label="FFT")
    ax.plot(Nx, RK4_exec_times, "bo", label="RK4")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"Execution time per step")
    fig.legend(loc="upper left", bbox_to_anchor=[0.2, 0.97], ncols=3, fontsize=10)
    fig.tight_layout()
    plt.savefig("../figures/AC1D_exec_time.png")

    print("Done with 1D analysis")

    return None


def analysis_2D():
    # 2D params
    Nx = [64, 128, 256]
    radius = 20
    L = 50
    w = 1.0
    H = 0.0
    tau = 1.0
    end_t = 10.0
    fe_times = []
    fe_errors = []
    fe_exec_times = []
    fft_times = []
    fft_errors = []
    fft_exec_times = []
    RK4_times = []
    RK4_errors = []
    RK4_exec_times = []

    for i, N in enumerate(Nx):
        print(10 * "-" + "1D, N = {} ".format(N) + 10 * "-")
        dt = (0.005 / (50 / 64)) * (L / N)
        AC2D = AllenCahn2D(L, N, radius, w, H, tau)

        # Forward Euler
        fe_time = time.time()
        tmp_times, tmp_errors, tmp_exec_times = AC2D.integrate(AC2D.FE_step, dt, end_t)
        print("done with FE, exec time = {:0.3f}".format(time.time() - fe_time))
        fe_times.append(tmp_times), fe_errors.append(tmp_errors), fe_exec_times.append(
            tmp_exec_times
        )

        # Reinitialize
        AC2D.__init__(L, N, radius, w, H, tau)

        # FFT
        AC2D.fft_init()
        fft_time = time.time()
        tmp_times, tmp_errors, tmp_exec_times = AC2D.integrate(AC2D.fft_step, dt, end_t)
        print("done with fft, exec time = {:0.3f}".format(time.time() - fft_time))
        fft_times.append(tmp_times), fft_errors.append(
            tmp_errors
        ), fft_exec_times.append(tmp_exec_times)

        # Reinitialize
        AC2D.__init__(L, N, radius, w, H, tau)

        # RK4
        RK4_time = time.time()
        tmp_times, tmp_errors, tmp_exec_times = AC2D.integrate(AC2D.RK4_step, dt, end_t)
        print("done with RK4, exec time = {:0.3f}".format(time.time() - RK4_time))
        RK4_times.append(tmp_times), RK4_errors.append(
            tmp_errors
        ), RK4_exec_times.append(tmp_exec_times)

    plot_errors(fe_times, fe_errors, Nx, usr_title="FE errors 2D", saveflag=1)
    plot_errors(fft_times, fft_errors, Nx, usr_title="FFT errors 2D", saveflag=1)
    plot_errors(RK4_times, RK4_errors, Nx, usr_title="RK4 errors 2D", saveflag=1)

    times = fe_times
    errors = fe_errors
    for i in range(len(Nx)):
        times.append(fft_times[i])
        errors.append(fft_errors[i])

    for i in range(len(Nx)):
        times.append(RK4_times[i])
        errors.append(RK4_errors[i])
    compare_errors(
        times,
        errors,
        Nx,
        usr_labels=["FE", "FFT", "RK4"],
        saveflag=1,
        usr_title="AC2D error comparison",
    )

    # execution times
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(Nx, fe_exec_times, "g*", label="FE")
    ax.plot(Nx, fft_exec_times, "r^", label="FFT")
    ax.plot(Nx, RK4_exec_times, "bo", label="RK4")

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"Execution time per step")
    fig.legend(loc="upper left", bbox_to_anchor=[0.2, 0.97], ncols=3, fontsize=10)
    fig.tight_layout()
    plt.savefig("../figures/AC2D_exec_time.png")

    print("Done with 2D analysis")
    return None


def main():
    analysis_1D()
    analysis_2D()
    return None


if __name__ == "__main__":
    main()
