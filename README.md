# Simulation Testbed for Liquid Chromatography
The Simulation Testbed for Liquid Chromatography (STLC) package implements variations of the general rate model. The aim is to provide a basis for the development of an easy to use package for chromatography modeling in python. 


## Ongoing development
---

This is research software that is still under development.


## Install
---
You can install by cloning repo and run:

```
	> pip install setuptools
	> python setup.py sdist
	> pip install -e ./
```

from project the root.

## Usage
---

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


## License
---

Released under GPL v3. License (see LICENSE.txt):

Copyright (C) 2022 Sartorius

