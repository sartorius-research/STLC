import matplotlib.pyplot as plt
import numpy as np

from stlc import lkm

k = 1e3
epsilon = 0.4
D = 1e-5
u = .1
tmax = 40
a_0 = 0.5
a_1 = 1
b_0 = 0.05
b_1 = 0.1
c0_0 = 10
c0_1 = 10


def step(t: float, x0: float, x1: float):
    return float(t > x0 and t < x1)


# Build a parameter object for each component
parameters0 = lkm.ModelParameters(u=u,
                                  ep=epsilon,
                                  D=D,
                                  c0=c0_0,
                                  k=k,
                                  a=a_0,
                                  b=b_0,
                                  ip=lambda t: step(t, 0, 12))

parameters1 = lkm.ModelParameters(u=u,
                                  ep=epsilon,
                                  D=D,
                                  c0=c0_1,
                                  k=k,
                                  a=a_1,
                                  b=b_1,
                                  ip=lambda t: step(t, 0, 12))

n = 10
ne = 40
zl = 1.
dt = 0.1
timesteps = int(tmax / dt)

total_points = ne * (n + 2 - 1) + 1 - 2
plt.figure()

model_fem = lkm.LumpedKineticModel(n, ne, zl, [parameters0, parameters1])

y_fem = lkm.solve(model_fem, tmax, dt)
plt.plot(np.arange(timesteps) * dt,
         y_fem[model_fem.c_end - 1],
         'k-+',
         label='fem',
         markersize=0.1,
         linewidth=0.2)
plt.plot(np.arange(timesteps) * dt,
         y_fem[model_fem.q_end + model_fem.c_end - 1],
         'y--+',
         label='fem',
         markersize=0.1,
         linewidth=0.2)

plt.legend()
plt.xlabel("z[m]")
plt.ylabel("c[g/l]")
plt.savefig("lkm_multi.pdf")

