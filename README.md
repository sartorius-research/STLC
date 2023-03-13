# Simulation Testbed for Liquid Chromatography
The Simulation Testbed for Liquid Chromatography (STLC) package implements variations of the general rate model. The aim is to provide a basis for the development of an easy to use package for chromatography modeling in Python. 


## Ongoing development

Please be aware that this is software created as part of research and is provided as is.


## Install

You can install by cloning the repository and running:

```
	> pip install setuptools
	> python setup.py sdist
	> pip install -e ./
```

from the project root.

## Usage
Examples are provided in the examples directory. A model may also be instantiated and run as follows:

```python
	from stlc import lkm
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
	dt = 0.01
	timesteps = int(tmax / dt)
	model = lkm.LumpedKineticModel(n, ne, zl, [parameters0])
	y = lkm.solve(model, tmax, dt)
```
## Cite
```
@article{ANDERSSON2023108068,
title = {Numerical simulation of the general rate model of chromatography using orthogonal collocation},
author = {David Andersson and Rickard Sjögren and Brandon Corbett},
journal = {Computers & Chemical Engineering},
volume = {170},
pages = {108068},
year = {2023},
doi = {https://doi.org/10.1016/j.compchemeng.2022.108068},
}
```

## References
Code implementing Orthogoanl collocation discretization: Larry C. Young 2019, [Orthogonal collocation revisited](https://doi.org/10.1016/j.cma.2018.10.019), Computer Methods in Applied Mechanics and Engineering.

## License

Released under GPL v3. License (see LICENSE.txt):

Copyright (C) 2023 Sartorius

