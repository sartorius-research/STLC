import matplotlib.pyplot as plt

from stlc import grm


def main():
    cm = 1e-2
    minute = 60
    zl = 1.7 * cm  #[cm]
    rp = 0.004 * cm  #[cm]
    ec = 0.4
    ep = 0.333
    kf = 0.01 * cm  #[cm]
    Dax = 0.002 * cm**2 / minute  #[cm^2 min^-1]
    Dp = 3.003 * 1e-6 * cm**2 / minute  #[cm^2 min^-1]
    Ds = 0.  #[cm^2 min^-1]
    u = 0.6 * cm / minute  #[cm min^-1]
    ka = 2.5  #[s^-1]
    kd = 1  #[s^-1]
    tinj = 20 * minute  #[min]
    cinj = 1  #[mol /m^-3]
    tmax = 100 * minute  #[min]
    qm = 1.
    nr = 10
    nz = 20
    dt = 1

    def step(t: float, tinj: float) -> float:
        return float(t <= tinj)

    parameters0 = grm.ModelParameters(c0=cinj,
                                          Dax=Dax,
                                          Dp=Dp,
                                          Ds=Ds,
                                          ka=ka,
                                          kd=kd,
                                          kf=kf,
                                          qm=qm,
                                          ip=lambda t: step(t, tinj))

    model = grm.GeneralRateModel(u=u,
                                         ep=ep,
                                         ec=ec,
                                         zl=zl,
                                         rp=rp,
                                         nz=nz,
                                         nr=nr,
                                         component_parameters=[parameters0])
    sol = grm.solve(model, tmax, dt)

    plt.plot(sol.t / minute,
             sol.c[0, -1, :],
             'r-',
             linewidth=0.2,
             label='STLC')
    for i in range(sol.nr2):
        plt.plot(sol.t / minute, sol.cp[0, -1, i, :], 'r-.', linewidth=0.1)

    plt.legend()
    plt.grid()
    plt.savefig("grm_single_component.pdf")


if __name__ == '__main__':
    main()
