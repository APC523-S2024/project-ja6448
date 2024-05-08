# APC 523 - Project - Josh Arrington - Spring 2024
import numpy as np
from scipy.integrate import quad
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TABLEAU_COLORS

import time

plt.rcParams.update({"font.size": 12})


class AllenCahn1D:
    """
    Class to handle the 1D Allen-Cahn equation
    """

    def __init__(
        self,
        L: float,
        zero_loc: float,
        Nx: int,
        w: float,
        H: float,
        tau: float,
    ) -> None:
        # Dynamics parameters
        self.w = w  # interface width control parameter
        self.H = H  # strength of interface driving
        self.tau = tau  # time constant

        # discretization info
        self.Nx = Nx
        self.x = np.linspace(0, L, Nx + 1, endpoint=True)  # x discretization
        self.x_min = self.x.min()
        self.x_max = self.x.max()
        self.h = (L) / (Nx)  # width of cells in space

        # interface profile
        self.zero_loc = zero_loc
        self.init_phi = self.generate_initial_interface(zero_loc)
        self.phi_exact = self.init_phi.copy()
        self.phi = self.init_phi.copy()
        self.interface = self.zero_loc

        # interface velocity
        self.exact_vel = self.calc_exact_velocity()
        self.numerical_vel = 0.0

        # time information
        self.time = 0.0

        # errors
        self.L2_error = 0.0

        return None

    def generate_initial_interface(self, zero_loc):
        """
        Generate initial_interface
        """
        return np.tanh((1 / (np.sqrt(2) * self.w)) * (self.x - zero_loc))

    def generate_symmetric_initial_interface(self, zero_loc, width):
        """
        Generate initial_interface
        """
        return (
            np.tanh((1 / (np.sqrt(2) * self.w)) * (self.x - zero_loc))
            - np.tanh((1 / (np.sqrt(2) * self.w)) * (self.x - zero_loc - width))
        ) - 1

    def calc_exact_velocity(self):
        phi_prime_sq = (
            lambda x: (
                (1 / (np.sqrt(2) * self.w))
                * (1 - np.tanh((x - self.zero_loc) / (np.sqrt(2) * self.w)) ** 2)
            )
            ** 2
        )
        A, _ = quad(phi_prime_sq, -np.inf, np.inf)

        return 4 / 3 * self.H / self.tau / A

    def exact_soln(self):
        self.phi_exact = np.tanh(
            (1 / (np.sqrt(2) * self.w))
            * (self.x - self.exact_vel * self.time - self.zero_loc)
        )

    def L2_norm(self):
        self.exact_soln()
        self.L2_error = (
            len(self.x) * self.h * np.sum(np.abs(self.phi - self.phi_exact) ** 2)
        )

    def second_deriv_FD(self, phi):
        """
        5-point central finite difference stencil for the second derivative
        """
        tmp = np.pad(phi, 2, mode="edge")
        f_pp = (1 / (12 * self.h**2)) * (
            -tmp[4:] + 16 * tmp[3:-1] - 30 * tmp[2:-2] + 16 * tmp[1:-3] - tmp[:-4]
        )

        return f_pp

    def df_dphi(self, phi):
        """
        evaluates the analytical form of the variational derivative of the free energy
        """
        phi_2 = self.second_deriv_FD(phi)
        df_dphi = self.w**2 * phi_2 + phi - phi**3 - self.H * (1 - phi**2)
        return df_dphi

    def find_interface(self):
        self.interface = np.interp(0.0, self.phi, self.x)

    def FE_step(self, dt):
        """
        Use Forward-Euler to march time
        """
        # FE step
        phi_old = self.phi.copy()
        self.phi = (dt / self.tau) * (self.df_dphi(phi_old)) + phi_old

        # march time forward
        self.time += dt

        # calc velocity
        old_interface = self.interface
        self.find_interface()
        self.numerical_vel = (self.interface - old_interface) / dt

        return None

    def RK4_step(self, dt):
        """
        Use RK4 to march time
        """

        def RHS(phi):
            return (1 / self.tau) * (self.df_dphi(phi))

        phi_old = self.phi.copy()

        # RK4 stage 1
        u1 = phi_old.copy()
        k1 = RHS(u1)

        # RK4 stage 2
        u2 = phi_old + k1 * dt / 2
        k2 = RHS(u2)

        # RK4 stage 3
        u3 = phi_old + k2 * dt / 2
        k3 = RHS(u3)

        # RK4 stage 4
        u4 = phi_old + k3 * dt
        k4 = RHS(u4)

        # update phi
        self.phi = phi_old + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # march time forward
        self.time += dt

        # calc velocity
        old_interface = self.interface
        self.find_interface()
        self.numerical_vel = (self.interface - old_interface) / dt

        return None

    def fft_init(self):
        """
        setting Fourier coefficients
        Following work of Biner et al
        """
        # fft_kx = 2 * np.pi * fft.fftfreq(2 * (self.Nx + 1), self.h)
        fft_kx = 2 * np.pi * fft.fftfreq(2 * len(self.phi), self.h)

        fft_k2 = fft_kx**2  # square of fourier coefficients
        self.k2 = fft_k2.copy()

    def fft_step(self, dt):
        """
        do fft timestep, semi-implicit ala Binder et al.
        """
        # pad phi so that it is nicely periodic
        padded_phi = np.concatenate([self.phi, np.flip(self.phi)])

        # evaluate variational derivative of current phi
        vfvphi = self.varf_varphi(padded_phi)

        # fft phi and its bulk variational derivative
        fphi = fft.fft(padded_phi)
        fvf = fft.fft(vfvphi)

        # calc new fft phi
        fphi = (dt / self.tau * fvf + fphi) / (1 + dt / self.tau * self.w**2 * self.k2)

        # inverse fft phi to update phi
        self.phi = np.real(fft.ifft(fphi))[: len(self.phi)]

        # march time forward
        self.time += dt

        # calc velocity
        old_interface = self.interface
        self.find_interface()
        self.numerical_vel = (self.interface - old_interface) / dt

        return None

    def varf_varphi(self, phi):
        """
        evaluate variational derivative of the bulk free energy
        """
        t1 = phi
        t2 = phi**3
        t3 = self.H * (1 - phi**2)
        return t1 - t2 - t3

    def integrate(self, func, dt, end_t, gen_prof=0, prof_title=""):
        self.phi = self.init_phi
        times = np.arange(0, end_t + dt, dt)
        exec_times = np.zeros(len(times) - 1)
        errors = np.zeros_like(exec_times)

        if gen_prof != 0:
            interfaces_exact = np.zeros_like(times)
            interfaces_numerical = np.zeros_like(times)
            interfaces_exact[0] = np.interp(0.0, self.phi_exact, self.x)
            interfaces_numerical[0] = self.interface

            color_table = TABLEAU_COLORS
            j = 0
            ctable = []
            for c in color_table:
                ctable.append(c)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(
                self.x,
                self.phi_exact,
                label="Exact, t = {:0.0f}".format(self.time),
                linestyle="-",
                color=ctable[j],
            )
            ax.plot(
                self.x,
                self.phi,
                label="Approx, t = {:0.0f}".format(self.time),
                linestyle="--",
                color=ctable[j],
            )
            j += 1

        for i, t in enumerate(times[1:]):
            # integrate forward
            step_time = time.time()
            func(dt)
            exec_times[i] = time.time() - step_time

            # errors
            self.exact_soln()
            self.L2_norm()
            errors[i] = self.L2_error

            if gen_prof != 0:
                if np.isclose(t % 5.0, 0.0):
                    ax.plot(
                        self.x,
                        self.phi_exact,
                        label="Exact, t = {:0.0f}".format(self.time),
                        linestyle="-",
                        color=ctable[j],
                    )
                    ax.plot(
                        self.x,
                        self.phi,
                        label="Approx, t = {:0.0f}".format(self.time),
                        linestyle="--",
                        color=ctable[j],
                    )
                    j += 1

            if (i + 1) % 10 == 0:
                print(
                    "Done with step {} of {}, exec time = {:0.3E}, error = {:0.3E}".format(
                        i, len(times), exec_times[i], self.L2_error
                    )
                )

        if gen_prof != 0:
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$\phi$")
            fig.legend()
            ax.set_title(prof_title)
            fig.tight_layout()
            plt.savefig(
                "../figures/" + "_".join(prof_title.split(" ")) + "prof_evo.png"
            )
        return times, errors, np.average(exec_times)

    def plot_phi(self, usr_title=0, save_flag=0, compare=0):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.x, self.phi, label="numerical soln")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\phi$")
        if usr_title != 0:
            ax.set_title(usr_title)
        if compare != 0:
            self.exact_soln()
            ax.plot(self.x, self.phi_exact, label="exact")
            fig.legend()

        fig.tight_layout()
        if save_flag != 0:
            plt.savefig(usr_title + ".png")

        return fig, ax

    def plot_interface(self, times, interface):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(times, interface)
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Interface position, $x$")

        return fig, ax

    def sim_animation(self, dt, end_t, save=0, show=0, func="FE"):
        dt = dt
        end_t = end_t
        times = np.arange(dt, end_t + dt, dt)
        n_frames = len(times)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim([self.x_min, self.x_max])
        ax.set_ylim([-3, 3])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\phi$")
        (line2,) = ax.plot([], [], lw=2, label="numerical soln", linestyle="--")
        (line1,) = ax.plot([], [], lw=2, label="exact soln")
        time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)
        interface_txt = ax.text(0.02, 0.90, "", transform=ax.transAxes)
        vel_txt = ax.text(0.02, 0.85, "", transform=ax.transAxes)
        error_txt = ax.text(0.02, 0.80, "", transform=ax.transAxes)
        fig.legend()

        def init_anim():
            """
            initialize animation
            """
            line1.set_data([], [])
            line2.set_data([], [])
            time_txt.set_text("time = {:0.3f}".format(self.time))
            interface_txt.set_text("interface = {:0.2f}".format(self.interface))
            vel_txt.set_text("interface vel = {:0.3f}".format(self.numerical_vel))
            error_txt.set_text("")

            return line1, line2, time_txt, interface_txt, vel_txt, error_txt

        def animate(i):
            """
            animation step
            """
            if func == "FE":
                self.FE_step(dt)
            elif func == "fft":
                self.fft_step(dt)
            elif func == "RK4":
                self.RK4_step(dt)
            self.L2_norm()

            line1.set_data(self.x, self.phi_exact)
            line2.set_data(self.x, self.phi)
            time_txt.set_text("time = {:0.3f}".format(self.time))
            interface_txt.set_text("interface = {:0.2f}".format(self.interface))
            vel_txt.set_text("interface vel = {:0.3f}".format(self.numerical_vel))
            error_txt.set_text("error = {:0.3E}".format(self.L2_error))

            return line1, line2, time_txt, interface_txt, vel_txt, error_txt

        if save != 0:
            if func == "FE":
                savetitle = "../slides/AC1D_FE.mp4"
            else:
                savetitle = "../slides/AC1D_RK4.mp4"
            ani = animation.FuncAnimation(
                fig,
                animate,
                frames=n_frames,
                interval=1,
                blit=True,
                init_func=init_anim,
            )
            ani.save(
                savetitle,
                fps=int(n_frames / 10.0),
                extra_args=["-vcodec", "libx264"],
            )

        if show != 0:
            ani = animation.FuncAnimation(
                fig,
                animate,
                frames=n_frames,
                repeat=True,
                repeat_delay=1000,
                interval=0.5,
                blit=True,
                init_func=init_anim,
            )
            plt.show()

        return None

    def plot_energy(self):
        fig, ax = plt.subplots()
        phi = np.linspace(-1.5, 1.5, 1000)
        f = lambda phi, H: -1 / 2 * phi**2 + 1 / 4 * phi**4 + H * (phi - phi**3 / 3)
        ax.plot(phi, f(phi, 0.0), label="H = 0.0")
        ax.plot(phi, f(phi, 1.0), label="H = 1.0")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$f(\phi)$")
        ax.set_xlim([-1.2, 1.2])
        ax.set_title("Bulk free energy")
        fig.legend()
        fig.tight_layout()
        plt.savefig("../figures/energy_well.png")

        return None


def main():
    Nx = 64
    L = 50
    w = 1.0
    H = 1.0
    tau = 1.0
    # dt = 0.05 * ((L) / Nx) ** 2 / 2
    dt = 0.005
    end_t = 30.0
    zero_loc = 10.0
    AC1D = AllenCahn1D(L, zero_loc, Nx, w, H, tau)
    AC1D.fft_init()
    # fig, ax = AC1D.plot_phi()
    # ax.set_xlim([0, 20])
    # plt.gca().set_aspect("equal")
    # plt.show()
    # AC1D.integrate(AC1D.fft_step, dt, end_t)
    # # AC1D.__init__(L, Nx, w, H, zero_loc, tau)
    AC1D.sim_animation(dt, end_t, show=0, save=1, func="RK4")
    # AC1D.plot_energy()

    return None


if __name__ == "__main__":
    main()
