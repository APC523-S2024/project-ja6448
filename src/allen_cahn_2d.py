# APC 523 - Project - Josh Arrington - Spring 2024
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import griddata
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import viridis, Reds
import time
from functools import partial
from skimage.measure import find_contours

plt.rcParams.update({"font.size": 12})


class AllenCahn2D:
    """
    Class to handle the 2D Allen-Cahn equation
    """

    def __init__(
        self,
        L: float,
        N: int,
        radius: float,
        w: float,
        H: float,
        tau: float,
    ) -> None:
        # Dynamics parameters
        self.w = w  # interface width control parameter
        self.H = H  # strength of interface driving
        self.tau = tau  # time constant

        # discretization info
        self.Nx, self.Ny = N, N
        self.L = L
        self.x, self.y = np.meshgrid(
            np.linspace(0, self.L, self.Nx + 1),
            np.linspace(0, self.L, self.Ny + 1),
        )
        self.x_center = L / 2
        self.y_center = L / 2
        self.h = L / N

        # interface profile
        self.radius = radius
        self.init_phi = self.generate_initial_interface(radius)
        self.phi_exact = self.init_phi.copy()
        self.phi = self.init_phi.copy()

        # interface position
        self.interface = 0
        self.find_interface()

        # area of inner region
        self.area = 0
        self.calc_area()
        self.init_area = self.area
        self.area_exact = 0
        self.exact_area(0)

        # interface velocity
        self.numerical_vel = 0.0

        # time information
        self.time = 0.0

        # errors
        self.L2_error = 0.0

        return None

    def generate_initial_interface(self, radius):
        r_vals = np.sqrt((self.x - self.x_center) ** 2 + (self.y - self.y_center) ** 2)
        return -np.tanh((1 / (np.sqrt(2) * self.w)) * (r_vals - radius))

    def find_interface(self):
        fig, ax = plt.subplots()
        cont = ax.contour(self.x, self.y, self.phi, [0])
        cont = cont.get_paths()[0].vertices
        self.interface_x = cont[:, 0]
        self.interface_y = cont[:, 1]
        r_vals = np.sqrt(
            (self.interface_x - self.x_center) ** 2
            + (self.interface_y - self.y_center) ** 2
        )
        self.interface = np.mean(r_vals)
        r_std = np.std(r_vals)
        self.interface_err = r_std / np.sqrt(len(r_vals))

        plt.close(fig)

        return None

    def calc_area(self):
        self.area = np.pi * self.interface**2

    def exact_area(self, t):
        if not (np.isclose(self.H, 0)):
            print("H nonzero")
            return None

        else:
            self.area_exact = self.init_area - 2 * np.pi * t / self.tau

    def second_deriv_FD(self, phi):
        """
        5-point central finite difference stencil for the second derivative
        """
        tmp = np.pad(phi, (1, 1), mode="edge")
        f_pp = (1 / self.h**2) * (
            tmp[2:, 1:-1]
            + tmp[:-2, 1:-1]
            + tmp[1:-1, 2:]
            + tmp[1:-1, :-2]
            - 4 * tmp[1:-1, 1:-1]
        )

        return f_pp

    def df_dphi(self, phi):
        """
        evaluates the analytical form of the variational derivative of the free energy
        """
        phi_2 = self.second_deriv_FD(phi)
        df_dphi = self.w**2 * phi_2 + phi - phi**3 - self.H * (1 - phi**2)

        return df_dphi

    def FE_step(self, dt):
        """
        use Forward Euler to march time
        """
        # FE step
        phi_old = self.phi.copy()
        self.phi = (dt / self.tau) * (self.df_dphi(phi_old)) + phi_old

        # march time forward
        self.time += dt

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

        return None

    def fft_init(self):
        n_value = fft.fftfreq(len(self.phi), 1.0 / len(self.phi))
        kx = np.zeros_like(self.phi)
        ky = np.zeros_like(self.phi)
        for i in range(len(self.phi)):
            for j in range(len(self.phi)):
                kx[i, j] = 2.0 * np.pi * n_value[j] / self.L
                ky[i, j] = 2.0 * np.pi * n_value[i] / self.L

        fft_k2 = kx**2 + ky**2

        self.k2 = fft_k2.copy()

    def fft_step(self, dt):
        old_phi = self.phi.copy()

        # evaluate variation derivative of current phi
        vfvphi = self.varf_varphi(old_phi)

        # fft phi and its bulk variational derivative
        fphi = fft.fft2(old_phi)
        fvf = fft.fft2(vfvphi)

        # calc new fft phi
        num = dt / self.tau * fvf + fphi
        denom = 1 + dt / self.tau * self.w**2 * self.k2
        fphi = np.divide(num, denom)

        # inverse fft phi to update phi
        self.phi = np.real(fft.ifft2(fphi))

        # march time forward
        self.time += dt

        return None

    def varf_varphi(self, phi):
        t1 = phi
        t2 = phi**3
        t3 = self.H * (1 - phi**2)

        return t1 - t2 - t3

    def integrate(self, func, dt, end_t, save_frames=0):
        self.phi = self.init_phi
        times = np.arange(0, end_t + dt, dt)
        exec_times = np.zeros(len(times) - 1)
        errors = np.zeros_like(exec_times)
        if save_frames != 0:
            j = 1
            fig, _ = self.contour_plot_phi(usr_title="t = {:0.3f}".format(0))
            plt.savefig("figs_2D/movie_{:03d}.png".format(j))
            plt.close(fig)
            j += 1

        for i, t in enumerate(times[1:]):
            # integrate forward
            step_time = time.time()
            func(dt)
            exec_times[i] = time.time() - step_time

            # errors
            self.find_interface()
            self.calc_area()
            self.exact_area(self.time)
            errors[i] = np.abs(self.area - self.area_exact)

            if (i + 1) % 10 == 0:
                if save_frames != 0:
                    fig, _ = self.contour_plot_phi(usr_title="t = {:0.3f}".format(t))
                    plt.savefig("figs_2D/movie_{:03d}.png".format(j))
                    plt.close(fig)
                    j += 1
                print(
                    "Done with step {} of {}, exec time = {:0.3E}, error = {:0.3E}".format(
                        i, len(times), exec_times[i], errors[i]
                    )
                )

        return times, errors, np.average(exec_times)

    def contour_plot_phi(self, usr_title=0, save_flag=0, compare=0):
        fig, ax = plt.subplots(figsize=(6, 6))
        cf = ax.contourf(self.x, self.y, self.phi, np.linspace(-1, 1, 100))
        ax.plot(
            self.interface_x,
            self.interface_y,
            color="red",
            lw="2",
            label="r = {:0.3f} +- {:0.3f}".format(self.interface, self.interface_err),
        )
        fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fig.colorbar(cf, ax=ax)  # , ticks=np.linspace(-1, 1, 11, endpoint=True))
        if usr_title != 0:
            ax.set_title(usr_title)
        if compare != 0:
            pass

        fig.tight_layout()

        if save_flag != 0:
            plt.savefig(usr_title + ".png")

        return fig, ax

    def surface_plot_phi(self, usr_title=0, save_flag=0, compare=0):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(self.x, self.y, self.phi, linewidth=0, cmap=viridis)
        # surf.set_clim([-1, 1])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\phi$")
        ax.set_zticks([-1, 0, 1])
        ax.elev = 30.0
        ax.azim = -135.0
        ax.set_box_aspect(aspect=None, zoom=0.8)
        fig.colorbar(surf, ax=ax)  # , ticks=np.linspace(-1, 1, 11))

        if usr_title != 0:
            fig.suptitle(usr_title)

        if compare != 0:
            pass

        if save_flag != 0:
            plt.savefig(usr_title + ".png")

        fig.tight_layout()

        return fig, ax

    def sim_animation(self, dt, end_t, save=0, show=0, func="FE", plot="contour"):
        dt = dt
        end_t = end_t
        times = np.arange(dt, end_t + dt, dt)
        n_frames = len(times)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim([0, self.L])
        ax.set_ylim([0, self.L])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\phi$")
        cvals = np.linspace(-1, 1, 100)
        cf = ax.contourf(self.x, self.y, self.phi, cvals)
        plt.title("time = {:0.3f}".format(self.time))
        r_txt = ax.text(
            0.02,
            0.95,
            "",
        )
        (line1,) = ax.plot(self.interface_x, self.interface_y, color="red", lw=2)
        fig.colorbar(cf, ax=ax, ticks=np.linspace(-1, 1, 11, endpoint=True))

        def animate(i, cont):
            """
            animation step
            """
            if func == "FE":
                self.FE_step(dt)
            else:
                pass
            # self.find_interface()

            ax.clear()
            cf = plt.contourf(self.x, self.y, self.phi, np.linspace(-1, 1, 100))
            # r_txt.set_text(
            #     "r = {:0.1f} +/- {:0.1E}".format(self.interface, self.interface_err)
            # )
            # line1.set_data(self.interface_x, self.interface_y)
            plt.title("time = {:0.3f}".format(self.time))

            return cf  # , r_txt  # , line1

        if save != 0:
            ani = animation.FuncAnimation(
                fig,
                animate,
                frames=n_frames,
                interval=1,
                blit=True,
            )
            ani.save(
                "AC2D_FE.mp4",
                fps=int(n_frames / 10.0),
                extra_args=["-vcodec", "libx264"],
            )

        if show != 0:
            ani = animation.FuncAnimation(
                fig,
                partial(animate, cont=cf),
                frames=n_frames,
                repeat=True,
                repeat_delay=1000,
                interval=0.5,
            )
            plt.show()

        return None


def main():
    # Parameters
    N = 64
    L = 50
    radius = 20
    w = 1.0
    H = 1.0
    tau = 1.0
    dt = 0.005
    end_t = 10.0

    AC2D = AllenCahn2D(L, N, radius, w, H, tau)
    # AC2D.exact_area()
    # AC2D.fft_init()
    AC2D.integrate(AC2D.RK4_step, dt, end_t, save_frames=1)

    # plt.show()

    return None


if __name__ == "__main__":
    main()
