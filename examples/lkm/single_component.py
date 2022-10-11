""" Single component example
"""
import matplotlib.pyplot as plt
from stlc import lkm


def run_simulation():
    zl = 1.0
    epsilon = 0.4
    u = 0.29
    tmax = 20
    a = 0.85
    D = 1e-6
    k = 111.0
    c_0 = 1.0
    b = 1.0

    parameters0 = lkm.ModelParameters(u=u, ep=epsilon, D=D, c0=c_0, k=k, a=a, b=b, ip = lambda t: t<1.)

    n = 10
    ne = 10
    dt = 0.001
    timesteps = int(tmax / dt)
    model = lkm.LumpedKineticModel(n, ne, zl, [parameters0])
    y_fem = lkm.solve(model, tmax, dt)
    result = y_fem[model.q_idx : model.q_end, int(timesteps / 5)]
    fig, ax = plt.subplots()
    ax.plot(model.zs, result, f"r-", linewidth=0.5, label=f"stlc")
    fig.savefig("single_component.pdf")


if __name__ == "__main__":
    run_simulation()
