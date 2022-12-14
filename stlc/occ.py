# Code contained in this script for orthogonal collocation discretization published
# by Larry Young on https://www.tildentechnologies.com/index.html with minor edits here

import numpy as np
from enum import Enum
import math

ShortOK = True  # ok to use shortcut for ultraspherical


class Typ(Enum):
    """e.g. occ.Typ.Gauss.value = numerical value"""

    Generic = 0
    Gauss = 1
    Lobatto = 2
    Chebyshev = 3
    RadauR = 4
    RadauL = 5
    Cheb1 = 6


class Geom(Enum):
    """e.g. occ.Geom.Planar.value = numerical value"""

    Nonsymmetric = 0
    Planar = 1
    Cylindrical = 2
    Spherical = 3


def _Wshift(n, w, wbfac):
    """Convert nonsymmetric weights and points on [-1,1] to [0,1]."""
    wbfac = (2.0**n) * wbfac
    w[0, :] = (w[0, :] + 1.0) * 0.5  # X
    w[2, :] = 2.0 * w[2, :]  # Wb
    w[3, :] = 0.5 * w[3, :]  # Wq
    return (w, wbfac)


def _Deriv1(sym, xx, wb):
    """Calculate first derivative matrix."""
    nt = xx.size
    if sym:
        x = xx**2
    else:
        x = xx
    A = np.empty((nt, nt))
    for i in range(nt):
        px = x[i] - x
        px[i] = 1.0
        A[i, :] = wb / (px * wb[i])  # /#
        A[i, i] = 1.0 - np.sum(A[i, :])
    if sym:
        for i in range(nt):
            A[i, :] = 2.0 * xx[i] * A[i, :]
    return A


def _Deriv1odd(geom, sym, x, wb):
    """Calculate first derivative matrix for an odd function."""
    nt = x.size
    An = np.empty((nt, nt))
    A = _Deriv1(sym, x, wb)
    if sym:
        gx = float(geom)
        for i in range(nt):
            An[i, :] = x[i] * A[i, :] / x[:]
            An[i, i] = An[i, i] + gx / x[i]
    else:
        An = A  # in case of call for nonsymmetric case
    return An


def _Deriv2(geom, sym, x, wb):
    """Calculate 2nd derivative or Laplacian matrix."""
    A = _Deriv1(sym, x, wb)
    if sym:
        B = _Deriv2sym(geom, x, A)
    else:
        B = _Deriv2non(x, A)
    return B


def _Deriv2non(x, A):
    """Calculate 2nd derivative for nonsymmetric case."""
    nt = x.size
    B = np.empty((nt, nt))
    for i in range(nt):
        B[i, :] = x[i] - x
        B[i, i] = 1.0
        B[i, :] = 2.0 * A[i, :] * (A[i, i] - 1.0 / B[i, :])  # /#
        B[i, i] = 0.0
        B[i, i] = -np.sum(B[i, :])
    return B


def _Deriv2sym(geom, x, A):
    """Calculate 2nd derivative for symmetric case."""
    nt = x.size
    B = np.empty((nt, nt))
    gx = 0.5 * float(geom)
    xx = x * x
    for i in range(nt):
        B[i, :] = xx[i] - xx
        B[i, i] = 1.0
        B[i, :] = 2.0 * A[i, :] * (A[i, i] + gx / x[i] - 2.0 * x[i] / B[i, :])
        B[i, i] = 0.0
        B[i, i] = -np.sum(B[i, :])
    return B


def _Stiffness(geom, sym, x, wb, wq):
    """Calculate Stifffness matrix."""
    nt = x.size
    A = _Deriv1(sym, x, wb)
    B = _Deriv2(geom, sym, x, wb)
    C = np.empty((nt, nt))
    for i in range(nt):
        C[i, :] = -wq[i] * B[i, :]
    C[nt - 1, :] = C[nt - 1, :] + A[nt - 1, :]
    if not sym:
        C[0, :] = C[0, :] - A[0, :]
    return C


def _Lcoeff(x, xc, wb=None, sym=0):
    """Interpolating polynomials thru xc evaluated at x"""
    if sym:
        L = _Lcoefx(x * x, xc * xc, wb)
    else:
        L = _Lcoefx(x, xc, wb)
    return L


def _Lcoefx(x, xc, wb=None):
    """Interpolating polynomials thru xc evaluated at x"""
    #  Calculate values of the Lagrange interpolating polynomials:
    #     x  - points where values are desired (evaluation points)
    #     xc - the n interpolation points (collocation points)
    #     wb - barycentric weights
    #     L(i,j) - value jth interpolating polynomial at xx(i)
    #  example usage: given values yc at the n interpolation points
    #     y(x) = MatMul(L,yc) is the interpolant at x(:)
    nx = x.size
    nc = xc.size
    L = np.empty((nx, nc))
    if wb is None:
        wb = _WBcalc(xc)
    for k in range(nx):
        if np.min(abs(x[k] - xc[:])) > 0.0001:
            L[k, :] = np.prod(x[k] - xc) * wb / (x[k] - xc)  # /#
        else:  # long calculation
            px = np.empty(nc)
            pp = np.empty(nc)
            for i in range(nc):
                px[:] = x[k] - xc
                px[i] = 1.0
                pp[i] = px.prod()
            L[k, :] = pp * wb
    return L


def _WBcalc(x):
    """Calculate barycentric weight from product formula."""
    n = x.size
    wb = np.empty(n)
    px = np.empty(n)
    for i in range(n):
        px = x[i] - x
        px[i] = 1.0
        wb[i] = 1.0 / px.prod()  # /#
    return wb


def _MassMatrix(oc, func=None, nextra=1):
    nt = oc.nt
    if func is None:
        if not oc.sym and oc.ptyp == Typ.Lobatto.value:
            D = _MassMatL(oc.x, oc.wb, oc.shft)
        elif not oc.sym and oc.ptyp == Typ.Gauss.value:
            D = _MassMatG(oc.x, oc.wb, oc.shft)
        else:
            D = np.zeros((nt, nt))
            for i in range(nt):
                D[i, i] = oc.wq[i]
    elif oc.ptyp <= Typ.Lobatto.value:
        D = _MassMatFn(oc, func, nextra)
    else:
        D = np.zeros((nt, nt))
        fx = func(oc.x)
        for i in range(nt):
            D[i, i] = oc.wq[i] * fx[i]
    return D


def _MassMatFn(oc, func, nextra=1):
    """Calculate Galerkin or Moments mass matrix with f(x)."""
    # approximated using quadrature with nextra extra points
    sym = oc.sym
    nt = oc.nt
    n = oc.n
    shift = oc.shft
    Gauss = oc.ptyp == Typ.Gauss.value
    Lobatto = oc.ptyp == Typ.Lobatto.value
    nq = n + nextra  # interior points for quadrature
    oq = OrthColloc(nq, oc.ptyp, oc.geom, oc.shft)
    i1 = 1 - sym
    i2 = nq + i1
    Lq = oc.Lcoeff(oq.x[i1:i2])
    wq = func(oq.x)
    if Gauss and (not sym) and shift:
        wq[i1:i2] = (
            wq[i1:i2] * (oq.wq[i1:i2]) / (oq.x[i1:i2] * (1.0 - oq.x[i1:i2]))
        )  # /# moments shifted
    elif Gauss:
        wq[i1:i2] = wq[i1:i2] * (oq.wq[i1:i2]) / (1.0 - (oq.x[i1:i2]) ** 2)  # /#
    else:
        wq = wq * (oq.wq)  # Galerkin
    D = np.empty((nt, nt))
    for i in range(nt):
        for j in range(i, nt):
            D[i, j] = np.sum(wq[i1:i2] * Lq[:, i] * Lq[:, j])
            D[j, i] = D[i, j]
    if Gauss and (not sym) and shift:
        for i in range(nt):
            D[i, :] = D[i, :] * (oc.x[i]) * (1.0 - oc.x[i])
    elif Gauss:
        for i in range(nt):
            D[i, :] = D[i, :] * (1.0 - (oc.x[i]) ** 2)
    elif Lobatto and sym:
        D[n, n] = D[n, n] + wq[nq]  # end pt. weights
    elif Lobatto:
        D[0, 0] = D[0, 0] + wq[0]
        D[n + 1, n + 1] = D[n + 1, n + 1] + wq[nq + 1]
    return D


def _MassMatG(x, wb, shift):
    """Calculate moments-Gauss mass matrix f(x) = 1."""
    n = x.size - 2
    D = np.zeros((n + 2, n + 2))
    Cn = -1.0 / (2.0 * n + 1.0)  # /#
    Cd = -2.0 * n
    if shift:
        wx = Cn * x * (1.0 - x) * wb
    else:
        wx = 2.0 * Cn * (1.0 - x * x) * wb
    for i in range(n + 1):
        D[i, :] = wx[i] * wb
        D[i, i] = Cd * D[i, i]
    return D


def _MassMatL(x, wb, shift):
    """Calculate Galerkin-Lobatto mass matrix for f(x) = 1."""
    n = x.size - 2
    D = np.empty((n + 2, n + 2))
    Cn = float(-(n + 1)) / float((2 * n + 3) * (n + 2))  # /#
    Cn = Cn if shift else Cn * 8.0
    Cd = float(-2 * (n + 1))
    for i in range(n + 2):
        D[i, :] = Cn * wb[i] * wb
        D[i, i] = Cd * D[i, i]
    return D


# ------------- OrthColloc Class for calculating basic data -----------
class OrthColloc:
    "Class for Orthogonal Collocation & MWR basic data"

    def __init__(self, n, TType=Typ.Lobatto.value, Geometry=0, Shift=True):
        self.n = n
        self.ptyp = TType
        self.geom = Geometry
        self.sym = max(0, min(Geometry, 1))
        if self.sym:
            self.shft = True  # symmetric always shifted
        else:
            self.shft = Shift
        self.nt = self.n + 2 - self.sym
        self.abeta = jac_abeta(TType, Geometry)
        w, wbfac = jac_quadrature(n, self.abeta, Geometry)
        if self.shft and self.geom == 0:
            w, wbfac = _Wshift(n, w, wbfac)
        self.x = w[0, :]
        self.th = w[1, :]
        self.wb = w[2, :]
        self.wq = w[3, :]
        self.wbfac = wbfac

    def Xpoints(self):
        """Return points or roots of polynomial and end points."""
        return self.x

    def WeightQ(self):
        """Return quadrature weights."""
        return self.wq

    def WeightB(self):
        """Return barycentric weights."""
        return self.wb

    def WBfac(self):
        """Return scaling factor for barycentric weights."""
        return self.wbfac

    def Deriv1(self):
        """Calculate first derivative matrix."""
        A = _Deriv1(self.sym, self.x, self.wb)
        return A

    def Deriv1odd(self):
        """Calculate first derivative matrix for odd (antisymmetric) function."""
        An = _Deriv1odd(self.geom, self.sym, self.x, self.wb)
        return An

    def Deriv2(self):
        """Calculate second derivative or Laplacian matrix."""
        B = _Deriv2(self.geom, self.sym, self.x, self.wb)
        return B

    def Stiffness(self):
        """Calculate second derivative or Laplacian matrix."""
        C = _Stiffness(self.geom, self.sym, self.x, self.wb, self.wq)
        return C

    def Lcoeff(self, x):
        L = _Lcoeff(x, self.x, (self.wb) * (self.wbfac), self.sym)
        return L

    def MassMatrix(self, func=None, nextra=1):
        D = _MassMatrix(self, func, nextra)
        return D


def jac_abeta(t, g):
    """alpha, beta & other basic parameters given point type & geometry"""
    tdict = {1: [0, 0], 2: [1, 1], 3: [0.5, 0.5], 4: [1, 0], 5: [0, 1], 6: [-0.5, -0.5]}
    ab = tdict.get(t)  # tdict[t] throws error not None
    if ab is None:
        ab = [0, 0]
    if g > Geom.Nonsymmetric.value:
        ab[1] = (g - 2) / 2  # /#
    return ab


def jac_quadrature(n, abeta=[0, 0], geom=0, MaxIter=1):
    w, wbfac = jac_quad(n, abeta, geom, MaxIter)  # Python code
    return (w, wbfac)


def jac_quad(n, abeta=[0, 0], geom=0, MaxIter=1):
    global ShortOK
    g = max(0, geom)
    a = abeta[0]
    b = abeta[1]
    if (a == 0) and (g == Geom.Cylindrical.value):
        ShortOK = False  # expedient to handle Cylindrical Gauss
    elif (a == b) and (g == Geom.Nonsymmetric.value) and ShortOK:
        g = -1
    a, b, bs, bc = a_beta(abeta, n)
    x, p = jac_roots(n, [a, b, bs, bc])
    w, wbfac = jac_weight(n, g, [a, b, bs, bc], x, p)
    return (w, wbfac)


def a_beta(abeta=[0, 0], n=0):
    """returns alpha & 3 beta values for quadrature calcs"""
    na = len(abeta)
    a = abeta[0]  # supplied
    b = abeta[1]  # supplied
    bs = b  # shortcut b if a == b (n needed for calculation)
    bc = b  # use in calcs (= bs except when full calcs demanded)
    if na > 2:
        bs = abeta[2]
    elif a == b:
        bs = (2 * (n % 2) - 1) * 0.5  # -0.5 even, +0.5 odd (-0.5 if no n)
    bc = bs if ShortOK else b
    return (a, b, bs, bc)


def n_values(n, abeta=[0, 0]):
    """returns 3 n values needed for quadrature calculations"""
    # n - supplied number of interior points
    # abeta = supplied alpha & beta
    # nf - n for associated full polynomial
    # nx - n for shortcut polynomial, also number x values
    # nc - n to use in calculations
    a = abeta[0]
    b = abeta[1]
    ia = int(a * 2) + 1  # /# 0,1,2,3 for a=-0.5,0,+0.5,1
    ib = int(round(b * 2)) + 1  # /# 0,1,2,3
    nf = n + (1 - ib % 2) * (n + ib // 2)  # /# n, or 2n or 2n+1
    nx = nf // 2 if (ia == ib or ib % 2 == 0) else nf
    nc = nx if ShortOK else n
    return (nf, nx, nc)


def jac_weight(n, g, abeta, x, p):
    """return w(4,nw) containing x,theta,wb,wq & wbfac"""
    # nn - number of interior points
    # nx - number of x points
    # n - order of polynomial
    # ns - order of corresponding shortcut polynomial
    # a = abeta[0];   b = abeta[1];   bs = abeta[2]
    nf, nx, nc = n_values(n, abeta)
    a, b, bs, bc = a_beta(abeta, n)
    s = max(0, min(1, g))
    nw = n + 2 - s
    i0 = nw - 1 - nx  # index of 1st unique point in w array
    ord = FirstLT(np.array([50, 7]), nf) + 2
    ex, cw, wbfac = jac_weight_coef(n, s, abeta)
    c = jac_deriv2(nc, [a, bc])  # coefficient in derivative expression
    w = np.zeros((4, nw), dtype=float)
    rx = np.empty((2, nx), dtype=float)
    dw = np.empty((2, nx), dtype=float)
    iw = np.array([0, nx + 1 - s, nw - 1])
    w[:2, iw] = np.array([(-1.0, 0.0, 1.0), (np.pi, np.pi, 0.0)])
    w[2:, iw] = cw[:, 1:]  # left,center,right wb & wq
    q = p[0, :] / (p[1, :] * (1.0 - x[0, :] ** 2))  # /# P/((1-x^2)P')
    rx[0, :] = jac_weight_rx(ex[0, :], x[0, :])
    rx[1, :] = jac_weight_rx(ex[1, :], x[0, :])
    w[0, i0 : nw - 1] = x[1, :]  # x right of midpoint
    w[1, i0 : nw - 1] = np.arccos(x[1, :])
    dw[0, :] = jac_wt_ord(ex[0, :], c, x[0, :], q, ord)
    dw[1, :] = jac_wt_ord(ex[1, :], c, x[0, :], q, ord)
    w[2, i0 : nw - 1] = cw[0, 0] / (rx[0, :] * p[1, :] * (dw[0, :] + 1.0))  # /# Wb
    w[3, i0 : nw - 1] = cw[1, 0] / (rx[1, :] * p[1, :] * (dw[1, :] + 1.0)) ** 2  # /# W
    w = jac_wts_store(n, nx, g, w)
    return (w, wbfac)


def jac_wts_store(n, nx, g, w):
    s = max(0, min(1, g))
    nodd = n % 2
    nw = n + 2 - s  # total weights
    i0 = nw - nx - 1  # first calculated x stored in w array
    g0 = Geom.Nonsymmetric.value  # Nonsymmetric, full
    gcyl = Geom.Cylindrical.value  # cylindrical
    if g == gcyl and n > nx:  # Gauss-cylindrical, special processing
        w[0, i0 - 1] = 0.5
        w[1, i0 - 1] = 0.5 * np.pi
        w[0, :nx] = (1.0 - w[0, n - 1 : i0 - 1 : -1]) * 0.5  # x^2
        w[0, i0:nw] = (1.0 + w[0, i0:nw]) * 0.5
        w[2, :nx] = (
            w[2, n - 1 : i0 - 1 : -1] * w[0, :nx] / w[0, n - 1 : i0 - 1 : -1]
        )  # /#
        w[0, :nw] = np.sqrt(w[0, :nw])
        w[1, :nx] = np.pi - w[1, n - 1 : i0 - 1 : -1]
        w[3, :nx] = w[3, n - 1 : i0 - 1 : -1]
        if nodd == 0:
            w[2, :nx] = -w[2, :nx]
    elif g != g0:  # other symmetric and shortcut
        w[0, i0:nw] = np.sqrt((1.0 + w[0, i0:nw]) * 0.5)
        if g < g0:  # shortcut
            w[1, i0:] = w[1, i0:] * 0.5
    if n > nx and g <= g0:  # reflect ultraspherical
        w[0, 1 : nx + 1] = -w[0, n : i0 - 1 : -1]  # x
        w[1, 1 : nx + 1] = np.pi - w[1, n : i0 - 1 : -1]  # theta
        w[3, 1 : nx + 1] = w[3, n : i0 - 1 : -1]  # quadrature weights
        if nodd == 0:
            w[2, 0 : nx + 1] = -w[2, nw : i0 - 1 : -1]  # barycentric weights
        else:
            w[2, 0 : nx + 1] = w[2, nw : i0 - 1 : -1]
    return w


def jac_wt_ord(ex, c, x, q, ord):
    # Expansion of Weight function for [(1+x)^u*(1-x)^v]*P'
    dw = np.empty(x.size, dtype=float)
    d = np.zeros((3, 4), dtype=float)
    u = ex[0] * 0.5
    v = ex[1] * 0.5
    if ord > 1:
        d[0, 0] = v - u - c[1]  # v - u - (alpha-beta)
        d[0, 1] = v + u - c[0]  # v + u - (alpha+beta+2)
    if ord > 2:
        d[1, 0] = (v - u) * d[0, 0] - d[0, 1] - c[2]
        d[1, 1] = (v - u) * d[0, 1] + (u + v - 2.0) * d[0, 0]
        d[1, 2] = (u + v - 1.0) * d[0, 1] + c[2]
    if ord > 3:
        d[2, 0] = (
            (c[1] * d[0, 0] + d[1, 0]) * (v - u) - (c[2] + d[0, 1]) * c[1] - d[1, 1]
        )
        d[2, 1] = (
            (c[0] * d[0, 0] + c[1] * d[0, 1] + d[1, 1]) * (v - u)
            + (c[1] * d[0, 0] + d[1, 0]) * (v + u)
            - (c[2] + d[0, 1]) * c[0]
            - 2.0 * (c[1] * d[0, 0] + d[1, 2])
            - 4.0 * d[1, 0]
        )
        d[2, 2] = (
            (c[0] * d[0, 1] + d[1, 2]) * (v - u)
            + (c[0] * d[0, 0] + c[1] * d[0, 1] + d[1, 1]) * (u + v)
            - 2.0 * c[0] * d[0, 0]
            + (c[2] - d[0, 1]) * c[1]
            - 3.0 * d[1, 1]
        )
        d[2, 3] = (
            (c[0] * d[0, 1] + d[1, 2]) * (u + v)
            + (c[2] - d[0, 1]) * c[0]
            - 2.0 * d[1, 2]
        )
    d[1, :] = d[1, :] * 0.5
    d[2, :] = d[2, :] / 6.0  # /#
    if ord <= 1:
        dw = 0.0
    elif ord == 2:
        dw = q * (d[0, 0] + d[0, 1] * x)
    elif ord == 3:
        dw = q * (d[0, 0] + d[0, 1] * x + q * (d[1, 0] + x * (d[1, 1] + x * d[1, 2])))
    else:  # ord == 4
        dw = q * (
            d[0, 0]
            + d[0, 1] * x
            + q
            * (
                d[1, 0]
                + x * (d[1, 1] + x * d[1, 2])
                + q * (d[2, 0] + x * (d[2, 1] + x * (d[2, 2] + x * d[2, 3])))
            )
        )
    return dw


def Sfunc(x):
    a = np.array(
        [
            1.0,
            1.0 / 12.0,
            1.0 / 288.0,
            -139.0 / 51840.0,
            -571.0 / 2488320.0,
            163879.0 / 209018880.0,
        ]
    )
    z = 1.0 / x
    s = a[0] + z * (a[1] + z * (a[2] + z * (a[3] + z * (a[4] + z * a[5]))))
    return s


def GammaRatio(a, b, n):
    """GammaRatio(a,b,n) = gamma(n+a)/gamma(n+b)"""
    na = n + a
    nb = n + b
    r = na / nb  # /#
    zswitch = 60
    if a == b:
        gr = 1.0
    elif a == b + 1:
        gr = float(n + b)
    elif b == a + 1:
        gr = 1.0 / float(n + a)  # /#
    elif min(na, nb) < zswitch:  # based on next term in expansion (see Sfunc)
        zn = np.arange(1, n)  # 0,1,2,...,n-1
        gr = np.prod((zn + a) / (zn + b))  # /#
        gr = gr * math.gamma(a + 1) / math.gamma(b + 1)  # /#
    else:  # Stirling approximation good to 1e-16 for n > ~50 to 60
        r = r ** (na - 0.5)
        gr = math.exp(b - a) * (nb ** (a - b)) * r * Sfunc(na) / Sfunc(nb)  # /#
    return gr


def jac_ends(n, abeta=[0, 0], Monic=False, Shift=False):
    """calculates end point values of jacobi polynomial"""
    # ---------------------------------------------------
    # Calculates polynomial values at left and right ends
    # Could easily be modified to calculate end derivatives
    # Direct calculation used, because Gamma(>170) fails
    # ---------------------------------------------------
    p = np.ones(2)
    a = abeta[0]
    b = abeta[1]
    p[0:2] = GammaRatio(b, 0, n + 1) / math.gamma(b + 1)  # /#
    if a != b:
        p[1] = GammaRatio(a, 0, n + 1) / math.gamma(a + 1)  # /#
    if Monic:
        p = p / jac_lead_coef(n, abeta, Shift)  # /#
    p[0] = p[0] * (1 - 2 * (n % 2))  # sign
    return p


def jac_lead_coef(n, abeta=[0, 0], Shift=False):
    """return leading coefficient of Pn"""
    # best to breakdown to prevent overflows
    #  an = Gamma(2n+a+b+1)/[Gamma(n+1)Gamma(n+a+b+1)]
    #  an = Cn*Gamma(n+c+1)*Gamma(n+c+0.5)/[Gamma(n+1)*Gamma(n+2c+1)]
    # where: c = (a+b)/2 & Cn = [2^(n+a+b)]/sqrt(pi)
    ab = abeta[0] + abeta[1]
    c = ab * 0.5
    xn = float(n)
    if n <= 160:
        an = math.gamma(xn + c + 1.0) / (math.gamma(xn + ab + 1.0) * np.sqrt(np.pi))
        an = an * math.gamma(xn + c + 0.5) / math.gamma(xn + 1.0)
    else:
        an = (
            math.lgamma(xn + c + 1.0)
            - math.lgamma(xn + ab + 1.0)
            - math.log(np.sqrt(np.pi))
        )
        an = math.exp(an + math.lgamma(xn + c + 0.5) - math.lgamma(xn + 1.0))

    if Shift:
        an = an * ((2.0) ** (xn * 2.0 + ab))
    else:
        an = an * ((2.0) ** (xn + ab))
    return an


def jac_poly_rec_all(x, n, ar, Monic=False):
    """returns P_0 thru P_n calculated by recurrence"""
    # x -  values calculated
    # n - order of polyomial
    # ar - recursion coefficients
    nx = x.size
    p = np.empty((n + 1, nx), dtype=float)
    nonzero = np.any(ar[0, :])
    p[0, :] = 1.0
    if Monic:
        p[1, :] = x - ar[0, 0]
        if nonzero:
            for k in range(1, n):
                p[k + 1, :] = (x - ar[0, k]) * p[k, :] - ar[1, k] * p[k - 1, :]
        else:
            for k in range(1, n):
                p[k + 1, :] = x * p[k, :] - ar[1, k] * p[k - 1, :]
    else:
        p[1, :] = x * ar[2, 0] - ar[0, 0]
        if nonzero:
            for k in range(1, n):
                p[k + 1, :] = (x * ar[2, k] - ar[0, k]) * p[k, :] - ar[1, k] * p[
                    k - 1, :
                ]
        else:
            for k in range(1, n):
                p[k + 1, :] = x * ar[2, k] * p[k, :] - ar[1, k] * p[k - 1, :]
    return p


def jac_weight_coef(n, s, abeta):
    """exponents; quadrature coefficients; left,center,riht weights; Wb scalin"""
    # Note: 1st index is 0 & 1 for Wb & Wq
    # ex - exponents in r(x) function (see below)
    # cw[:,0] - constant in weight
    # cw[:,1-3] = weight for left end, center & right end
    #             some of these are not used & overwritten
    # wbfac - scaling factor in barycentric weights
    # Note: the weights are:
    #  w = cw[0]*[r(x)*p']^(-1 or -2)
    #  r(x) = (1+x)^u*(1-x)^v, where u = ex(0)/2 & v = ex(1)/2
    cw = np.zeros((2, 4), dtype=float)
    ex = np.zeros((2, 2), dtype=int)
    s0 = np.ones((2), dtype=float)
    nf, nx, nc = n_values(n, abeta)
    a, b, bs, bc = a_beta(abeta)
    fshft = 2**nc
    zn1 = n + 1
    zn8 = 8.0 * zn1 / (zn1 + 1)  # /#
    sq2 = math.sqrt(2)
    wbfac = jac_lead_coef(n, [a, b])  # wb normalizing factor
    p0 = jac_ends(n, [a, b])
    ib = round(bc * 2) + 2  # 1,2,3,4 for b = -0.5,0,+0.5,1
    ia = round(a)  # 0,1
    icase = (ia * 10 + ib) * (1 - 2 * s)  # 1 to 14 nonsymmetric, negative for symmetric
    Radau = icase == 4 or icase == 12
    if Radau:
        an = 1.0
    else:
        s0 = jac_ends(nx, [a, bs])
        an = p0[1] / s0[1]
        cw[0, 2] = -s0[1] / (s0[0] * p0[1])  # Wb center
    cw[0, 1] = -0.5 / p0[0]  # /# Wb ends
    cw[0, 3] = +0.5 / p0[1]  # /# Wb ends
    if icase == 2:  # /# Gauss full
        ex[0, :] = 2
        ex[1, :] = 1
        cw[0, 0] = -1.0
        cw[1, 0] = 2.0
        cw[1, 2] = 2.0 * cw[0, 2] ** 2
    elif icase == 1:  # /# Gauss shortcut even
        ex[0, :] = [1, 2]
        ex[1, :] = 1
        cw[0, 0] = -1.0 / (an * sq2)
        cw[1, 0] = 0.5
    elif icase == 3:  # /# Gauss shortcut odd
        ex[0, :] = 2
        ex[1, :] = [2, 1]
        cw[0, 0] = -1.0 / an
        cw[1, 0] = 1.0
        cw[1, 2] = 2.0 * cw[0, 2] ** 2
    elif icase == 14:  # /# Lobatto full
        ex[:, :] = 2
        cw[0, 0] = -1.0  # /#
        cw[1, :] = zn8 * cw[0, :] ** 2
    elif icase == 11:  # /# Lobatto shortcut even
        ex[:, :] = [1, 2]
        cw[0, 0] = -1.0 / (an * sq2)
        cw[1, :] = zn8 * cw[0, :] ** 2
    elif icase == 13:  # /# Lobatto shortcut odd
        ex[:, :] = 2
        cw[0, 0] = -1.0 / an
        cw[1, :] = zn8 * cw[0, :] ** 2
    elif icase == 12:  # /# Radau right
        ex[0, :] = 2
        ex[1, :] = [1, 2]
        cw[0, 0] = -1.0
        cw[1, 0] = 4.0
        cw[1, 3] = 2.0 / (p0[1] * p0[1])
    elif icase == 4:  # /# Radau left
        ex[0, :] = 2
        ex[1, :] = [2, 1]
        cw[0, 0] = -1.0
        cw[1, 0] = 4.0
        cw[1, 1] = 2.0 / (p0[0] * p0[0])
    elif icase >= -10:  # /# Gauss symmetric
        ex[0, :] = [0, 2]
        ex[1, :] = 1
        cw[0, 0] = -1.0
        cw[0, 3] = (1.0) / p0[1]
        cw[1, 0] = 0.5
        cw[1, 2] = (0.5) * cw[0, 2] ** 2
        wbfac = wbfac * fshft
    elif icase >= -20:  # /# Lobatto symmetric
        ex[0, :] = [0, 2]
        ex[1, :] = [1, 2]
        cw[0, 0] = -1.0
        cw[0, 3] = (1.0) / p0[1]
        cw[1, 0] = zn1 / (zn1 + b)
        cw[1, 3] = cw[1, 0] / (p0[1] * p0[1] * 2.0)  # /#
        wbfac = wbfac * fshft
    return (ex, cw, wbfac)


def jac_weight_rx(ex, x):
    """calculate r(x) = [(1+x)**ex1/2]*[(1-x)**ex2/2]"""
    case = ex[0] * 10 + ex[1]
    if case == 2:
        rx = 1.0 - x
    elif case == 11:
        rx = np.sqrt(1.0 - x * x)
    elif case == 12:
        rx = np.sqrt((1.0 - x) * (1.0 - x * x))
    elif case == 21:
        rx = np.sqrt((1.0 + x) * (1.0 - x * x))
    elif case == 22:
        rx = 1.0 - x * x
    return rx


def jac_roots(n, abeta, MaxIter=1, Order=None, Theta=False, nneg=0):
    """Calculate roots of Jacobi Polynomial with higher order iteration method"""
    nf, nx, nc = n_values(n, abeta)
    a, b, bs, bc = a_beta(abeta, n)
    x = np.empty((2, nx), dtype=float)
    x[0, :] = jac_root_estimate(nx, abeta, nneg=False)
    x[0, :] = np.cos(x[0, :])
    ib = int(round(bs * 2)) + 1  # /# 0,1,2,3
    # nf = nx + (1-ib%2)*(nx + ib//2)  #/# n, or 2n or 2n+1
    # n = nx if ShortOK else nf
    Order = FirstLT(np.array([35, 9]), nf) + 1
    c2 = jac_deriv2(nc, [a, bc])  # coefficient in derivative expression
    p = jac_poly_recurrent(x[0, :], nc, [a, bc], nd=1)  # calculate p & p'
    dp = jac_roots_iter(x[0, :], p, c2, Order)  # calculate correction to x
    x[1, :] = x[0, :] - dp  # incorrect if nneg > 0
    # p = jac_poly_taylor(n,abeta,xn,dp,p,c2,OrdP)   # improve p & p' with Taylor series
    return (x, p)


def jac_roots_iter(x, p, c2, Order):
    """calculate higher order root change"""
    # calculate 1 or 2 terms beyond Newton-Raphson
    dp = p[0, :] / p[1, :]  # /
    if Order > 1:
        nx = x.size
        dx = np.empty((Order - 1, nx), dtype=float)
        r2 = 1.0 - x * x
        dpx = dp / r2  # /#
        dx[0, :] = 0.5 * (c2[0] * x + c2[1])  # Order=2, 3rd order convergence
    if Order > 2:  # Order = 3, 4th order
        d30 = ((c2[2] + c2[1] * c2[1]) * 2.0 - c2[0]) / 6.0  # /#
        d31 = c2[1] * (c2[0] - 0.5) * 4.0 / 6.0  # /#
        d32 = (c2[0] * c2[0] - c2[0] * 0.5 - c2[2]) * 2.0 / 6.0  # /#
        dx[1, :] = d30 + x * (d31 + x * d32)
    if Order == 2:
        dp = dp * (1.0 + dpx * dx[0, :])
    elif Order == 3:
        dp = dp * (1.0 + dpx * (dx[0, :] + dpx * dx[1, :]))
    return dp


def jac_poly_recurrent(
    x, n, abeta=[0, 0], nd=0, n0=0, Monic=False, Shift=False, Shortcut=False
):
    """calculate Pn and optionally dPn,Pn-1 & Pn-2 by recurrence"""
    # nd = number of derivatives, 0 or 1
    # n0 = number of lower polynomials, 0, 1 or 2
    p = jac_poly_rec(x, n, abeta, nd, n0, Monic, Shift)
    # *** add calculation from shortcut ***
    return p


def jac_poly_rec(x, n, abeta=[0, 0], nd=0, n0=0, Monic=False, Shift=False):
    """returns polynomials pn(x) & nd derivatives calculated with recurrence"""
    # x - x values
    # n - highest degree
    # abeta - alpha & beta
    # nd - number of derivatives Pn', Pn",...
    # n0 - number preceeding - Pn-n0, Pn-n0+1,...,Pn
    # returns Pn-2,Pn-1,Pn,Pn',Pn",... 1 + n0 + nd arrays
    nx = x.size
    id = n0 + nd  # index of 1st derivative
    p = np.empty((n0 + nd + 1, nx))
    ar = jac_reccoef(n + 1, abeta, Monic, Shift)
    nonzero = not np.any(ar[0, :])
    if n0 < 3:
        i0 = 2  # index of Pn
        p3 = jac_poly_rec3(x, n, ar, Monic)
    else:
        i0 = n  # index of Pn
        p3 = jac_poly_rec_all(x, n, ar, Monic)
    p[0 : n0 + 1, :] = p3[i0 - n0 : i0 + 1, :]  # copy polynomials requested
    if nd > 0:  # calculate 1st derivative
        if not Shift:
            xx = np.clip((1.0 - x * x), a_min=1.0e-20, a_max=1.0)
        else:
            xx = np.clip((x * (1.0 - x)), a_min=1.0e-20, a_max=1.0)
        c = jac_deriv1(n, abeta, Monic, Shift)
        id = n0 + 1  # index of 1st derivative
        p[id, :] = ((c[0] * x + c[1]) * p3[i0, :] + c[2] * p3[i0 - 1, :]) / xx  # /#  p'
        for i in range(0, nd - 1):
            c = jac_deriv2(n, abeta, Shift, nd=i + 2)
            p[id + i + 1, :] = (
                (c[0] * x + c[1]) * p[id + i, :] + c[2] * p[id + i - 1, :]
            ) / xx  # /# p" & higher
    return p


def jac_poly_rec3(x, n, ar, Monic=False):
    """returns Pn,Pn-1 & Pn-2 calculated by recurrence"""
    #  x -  values calculated
    #  n - order of polyomial
    #  ar - recursion coefficients''
    nx = x.size
    p = np.empty((3, nx), dtype=float)
    kx = np.array([(2, 0, 1), (1, 2, 0), (0, 1, 2)])
    k = n % 3
    k0 = kx[k, 0]  # logic so ending order is n-2,n-1 & n
    k1 = kx[k, 1]
    k2 = kx[k, 2]
    # ar0_zero = ar.any(  # check for all zeros
    p[k0, :] = 1.0
    if Monic:
        p[k1, :] = x - ar[0, 0]
        for k in range(1, n):
            p[k2, :] = (x - ar[0, k]) * p[k1, :] - ar[1, k] * p[k0, :]
            k3 = k0
            k0 = k1
            k1 = k2
            k2 = k3
    else:
        p[k1, :] = x * ar[2, 0] - ar[0, 0]
        for k in range(1, n):
            p[k2, :] = (x * ar[2, k] - ar[0, k]) * p[k1, :] - ar[1, k] * p[k0, :]
            k3 = k0
            k0 = k1
            k1 = k2
            k2 = k3
    return p


def jac_deriv1(n, abeta=[0, 0], Monic=False, Shift=False):
    """return c in: (1-x^2)Pn' = (c[0]*x + c[1])Pn + c[2]*Pn-1"""
    a = abeta[0]
    b = abeta[1]
    c = np.empty(3, dtype=float)
    xn = float(n)
    t = 1.0 / (2.0 * xn + a + b)  # /
    c[0] = -xn
    c[1] = xn * (a - b) * t
    c[2] = 2.0 * (xn + a) * (xn + b) * t
    if Shift:  # convert for shifted polynomial
        c[1] = (c[1] - c[0]) * 0.5
        c[2] = c[2] * 0.5
    if Monic:  # convert to Monic
        c[2] = (
            c[2] * jac_lead_coef(n - 1, abeta, Shift) / jac_lead_coef(n, abeta, Shift)
        )  # /
    return c


def jac_reccoef(n, abeta=[0, 0], Monic=False, Shift=False):
    """return recursion coefficient for Jacobi polynomial"""
    # Function to calculate the recursion coefficients of a Jacobi polynomial
    # Monic = true/false for monic or conventional form
    # Shift = true/false shifted (0,1) vs not shifted (-1,1)
    # The coefficients are in a 4xn array, where:
    # Monic form gives
    #  P_k+1 = (x - ar[0,k]*P_k - ar[1,k]*Pn-1
    #  ar[2,:] - the leading coefficient of conventional form
    # Conventional form (Monic = False) give coefficients:
    #  P_k+1 = (x*ar[2,k] - ar[0,k])*P_k - ar[1,k]*P_k-1
    # In either case:
    #  ar[3,k] - Integral(Pn^2*w(x))
    a = abeta[0]
    b = abeta[1]
    fac2 = 2.0
    if Shift:
        fac2 = 1.0
    a1 = a + 1
    b1 = b + 1
    ab1 = a1 + b
    b2a2 = b * b - a * a
    ab = a + b
    gab1 = math.gamma(a1) * math.gamma(b1)
    ar = np.empty((4, n))
    r = np.empty(n + 1)
    zn = np.arange(n + 1)
    z2nab = np.empty(n + 1)
    z2nab = 2 * zn + ab
    r[0] = (a1 + b1) / fac2
    r[1:] = (
        (2.0 * zn[1:] + a1 + b1)
        * (2 * zn[1:] + ab1)
        / ((zn[1:] + 1) * (zn[1:] + ab1) * fac2)
    )
    ar[0, 0] = (b - a) / (ab + 2.0)
    ar[1, 0] = ((2.0) ** (ab + 1.0)) * gab1 / math.gamma(a1 + b1)
    ar[2, 0] = 1.0
    ar[3, 0] = gab1 * (fac2**ab1) / math.gamma(ab1 + 1.0)
    if n > 1:
        ar[0, 1:] = b2a2 / (z2nab[1:n] * (z2nab[1:n] + 2.0))
        ar[1, 1] = 4.0 * a1 * b1 / ((ab + 3.0) * (a1 + b1) ** 2)
        ar[1, 2:] = (
            4.0
            * zn[2:n]
            * (zn[2:n] + a)
            * (zn[2:n] + b)
            * (zn[2:n] + ab)
            / ((z2nab[2:n] + 1.0) * (z2nab[2:n] - 1.0) * (z2nab[2:n] ** 2))
        )
    if Shift:
        ar[0, :] = (1.0 + ar[0, :]) * 0.5
        ar[1, :] = ar[1, :] * 0.25
    for k in range(1, n):
        ar[2, k] = ar[2, k - 1] * r[k - 1]
        ar[3, k] = ar[3, k - 1] * ar[1, k] * r[k - 1] ** 2
    if not Monic:
        am = np.empty((2, n))
        am[0, :] = ar[0, :] * r[0:n]
        am[1, 1:] = ar[1, 1:] * r[1:n] * r[0 : n - 1]
        am[1, 0] = a * b * (a1 + b1)
        if (ab1 * ab) > 0.0:
            am[1, 0] = am[1, 0] / (ab1 * ab)  # /
        ar[2, :] = r[0:n]
        ar[:2, :] = am
    return ar


def jac_root_estimate(nx, abeta=[0, 0], nneg=False):
    """estimate roots of nth Jacobi polynomial,returns theta = arccos(x)"""
    #  Estimate roots of Jacobi polynomials using up to 4 different
    #  approximations (no more than 2 for a given set of roots).
    #  It is optimized for Gauss, Lobatto & Radau base points.
    #  Cutoff values are used to chose between boundary & interior methods.
    #   i1:i2 - range of points using interior correlation
    #   xe - a cutoff value between boundary and interior correlation
    #   ae - cutoff angle arccos(xe)
    #   bmeth - boundary method to use
    #  ------------------------------------------------
    if nx == 0:
        return None
    a, b, bs, bc = a_beta(abeta)
    ba = b
    i1 = 1
    bmeth = 2
    ib = round(bc * 2)
    Shortcut = abs(ib) == 1
    ib = round(bs * 2)
    Ultraspherical = abs(ib) == 1
    Legendre = a == 0 and Ultraspherical
    Lobatto = a == 1 and Ultraspherical
    n = nx
    if Ultraspherical:
        ba = a
        abetax = [a, a]
        n = round(2 * nx + (ib + 1) / 2)  # /# full
    i2 = nx
    zn = float(n)
    x = np.empty(nx, dtype=float)
    an = np.arange(2 * (nx - 1), -1, -2)
    an = np.pi * (an + a + 1.5) / (2 * n + 1 + a + ba)  # /# rough estimate for cutoffs
    if Legendre:  # Legendre a=b=0
        if n < 19:  # linear
            xe = 0.50 + zn * 0.019
            bmeth = 1
        else:  # large n, bmeth = 2
            xe = 0.57 - zn * (0.001)
            xe = max(xe, 0.480)
    elif Lobatto:  # Ultraspherical - Lobatto
        xe = max(0.780, 0.813 - zn * 0.00075)
        if n < 5:
            xe = 0.60  # small n
        if n < 20:
            bmeth = 1  # large n, change boundary method
    elif Ultraspherical:  # Ultraspherical - not Legendre or Lobatto
        x = an  # use crude estimate
        return x
    else:  # Radau - boundary method left & right ends
        # # xe = -0.5;   ae = acos(xe);  ix = FirstLT(a0,ae)-1;  #uncomment to use
        abetax = [b, a]
        xe = 0.10 * (b - a)
        ae = math.acos(xe)
        ix = FirstLT(an, ae)
        if ix > 0:
            x[n - 1 : n - ix - 1 : -1] = np.pi - jac_root_est_boundary(
                n, ix, abetax, bmeth
            )
        xe = -0.8
        # a positive value will use interior method in center
        i2 = n - ix
        abetax = abeta
    ae = math.acos(xe)
    i1 = min(i2, nx - FirstLT(an, ae))  # no. boundary points
    if n < 4:
        x = jac_root_stored(n, Legendre, Lobatto, abetax)
    else:
        x[0:i1] = jac_root_est_boundary(n, i1, abetax, bmeth)
        x[i1:i2] = jac_root_est_interior(n, i1, i2, abetax)
    if Shortcut:
        x = x * 2.0
    xr = x[::-1]  # reverse values
    if nneg:
        ng = FirstLT(xr, 0.5 * np.pi)
        xr[:ng] = np.pi - xr[:ng]
        return (xr, ng)
    else:
        return xr


def FirstLT(x, x0):
    """returns k for first x[k] < x0"""
    m = x.size
    for k in range(m):
        if x[k] < x0:
            break
    else:
        k = m  # k = m -1 if no break
    return k


def jac_root_stored(n, Legendre, Lobatto, abeta):
    """returned stored roots for n < 4"""
    if Legendre or Lobatto:
        if n > 1:
            x = np.empty(1)
        else:
            return None
    if Legendre:
        Groots = np.array([0.955316618124509278, 0.684719203002282914])
        x[0] = Groots[n - 2]
    elif Lobatto:
        Lroots = np.array([1.107148717794090503, 0.857071947850130988])
        x[0] = Lroots[n - 2]
    else:
        Rroots = np.array(
            [
                1.910633236249018556,
                1.276676121181329395,
                2.332144396859698049,
                0.957802316375337495,
                1.752866861904567592,
                2.537158998077295037,
            ]
        )
        i1 = max(0, 2 * n - 3)
        xr = Rroots[i1 : i1 + n]
        if abeta[0] > abeta[1]:
            x = xr
        else:
            x = np.pi - xr[::-1]
    return x


def jac_root_est_interior(n, i1, i2, abeta=[0, 0], Method=2):
    """estimate roots i1 - i2 for nth Jacobi polynomial using interior methods"""
    #  1. Gatteschi and Pittaluga [see Gautschi & Giordano, Hale & Townsend]
    #  2. Tricomi valid only for Legendre case.
    # ---------------------------------------------------------------------------
    nx = i2 - i1 + 1
    if nx < 1:
        return None
    a = abeta[0]
    b = abeta[1]
    zn = float(n)
    x = np.empty(nx, dtype=float)
    Legendre = (a == 0) and (b == 0)
    if not Legendre:
        Method = 1
    if Method == 1:  # general interior method
        r = 1.0 / (2.0 * zn + a + b + 1.0)  # /
        x = np.arange(2.0 * (i1 + 1), 2.0 * (i1 + nx), 2.0)
        x = (x + a - 0.5) * r * np.pi
        d0 = np.tan(0.5 * x)
        d0 = ((0.25 - a * a) / d0 - (0.25 - b * b) * d0) * r * r  # /
        x = x + d0
    else:  # Legendre - valid for and best method
        x = np.arange(float(i1 + 1), float(i1 + nx))
        xn1 = zn - 1.0
        xn4 = 1.0 / zn**4  # /
        z1 = 1.0 - xn4 * (0.125 * zn * xn1 + (39.0 / 384.0))  # /
        x = np.pi * (x - 0.25) / (zn + 0.5)  # /
        d0 = xn4 * (28.0 / 384.0) / (np.sin(x) ** 2)  # /
        x = (z1 + d0) * np.cos(x)
        x = np.arccos(x)
    return x


def jac_root_est_boundary(n, nx, abeta=[0, 0], Method=2):
    """estimate 1st nx roots for nth Jacobi polynomial, boundary methods"""
    # Boundary Estimation methods. Using Bessel zeroes
    #  1. Gatteschi [Gautschi and Giordano (2008)]
    #  2. Olver, more uniform and better for x < 0.80 (appx.)
    # ---------------------------------------------------------------------------
    if nx < 1:
        return None
    a = abeta[0]
    b = abeta[1]
    a2 = a * a
    b2 = b * b
    r = float(2 * n) + a + b + 1.0
    x = jac_root_est_bessel(nx, a)
    if Method == 1:
        c33 = 1.0 / 3.0
        c45 = 1.0 / 45.0
        v = 1.0 / np.sqrt(r * r + c33 - c33 * a2 - b2)  # /#
        r = c45 * (4.0 - a2 - 15.0 * b2) * v**4
        xcot = 2.0 * x * v
        x = 2.0 * x * v * (1.0 - r * (0.5 * x**2 + a2 - 1.0))
    else:
        r = 1.0 / r  # /#
        b2 = a2 - b2
        a2 = 2.0 * a2 - 0.5
        x = x * r * 2.0
        xtan = np.tan(x * 0.5)
        xcot = a2 * (1.0 / x - 0.5 * (1.0 / xtan - xtan))
        xcot = (xcot - b2 * xtan) * r * r
        x = x + xcot
    return x


def jac_deriv2(n, abeta=[0, 0], Shift=False, nd=2):
    """return c in: (1-x^2)P'' = (c[0]*x + c[1])Pn' + c[2]*Pn"""
    # Coefficients valid for monic or conventional Pn
    a = abeta[0]
    b = abeta[1]
    c = np.empty(3, dtype=float)
    xn = float(n)
    c[0] = 2.0 + a + b
    c[1] = a - b
    c[2] = -xn * (xn + a + b + 1.0)
    for j in range(3, nd + 1):  # coefficients for 3rd and higher
        c[2] = c[2] + c[0]
        c[0] = c[0] + 2.0
    if Shift:  # modify if shifted
        c[1] = (c[1] - c[0]) * 0.5
    return c


def jac_root_est_bessel(n, a):
    """returns n roots Bessel function of kind "a" = 0 or 1"""
    #  16 stored, others calculated using 5 terms of McMahon's approximation
    #  roots to double precision accuracy, i.e. ~2e-16 maximum error
    # ---------------------------------------------------------------------------------------------
    nstored = 16
    ia = int(round(a))
    if ia == 0:
        Jbessel = np.array(  # stored roots accurate to 28 digits
            [
                2.40482555769577276862163188,
                5.52007811028631064959660411,
                8.65372791291101221695419871,
                11.79153443901428161374304491,
                14.93091770848778594776259400,
                18.07106396791092254314788298,
                21.21163662987925895907839335,
                24.35247153074930273705794476,
                27.49347913204025479587728823,
                30.63460646843197511754957893,
                33.77582021357356868423854635,
                36.91709835366404397976949306,
                40.05842576462823929479930737,
                43.19979171317673035752407273,
                46.34118837166181401868578888,
                49.48260989739781717360276153,
            ]
        )
    else:
        Jbessel = np.array(  # stored roots accurate to 28 digits
            [
                3.83170597020751231561443589,
                7.01558666981561875353704998,
                10.17346813506272207718571178,
                13.32369193631422303239368413,
                16.47063005087763281255246047,
                19.61585851046824202112506588,
                22.76008438059277189805300515,
                25.90367208761838262549585545,
                29.04682853491685506664781988,
                32.18967991097440362662298410,
                35.33230755008386510263447902,
                38.47476623477161511205219756,
                41.61709421281445088586351681,
                44.75931899765282173277935271,
                47.90146088718544712127400872,
                51.04353518357150946873303463,
            ]
        )
    j = min(n, nstored)
    if n <= nstored:
        return Jbessel[:n]
    j = nstored  # use McMahons approximation for the rest
    j0 = np.empty(n, dtype=float)
    j0[:j] = Jbessel
    constf = np.array([1.0 / 6.0, 1.0 / 30.0, 0.125 / 105.0])  # /
    const2 = np.array([-31.0, 7.0])
    const3 = np.array([3779.0, -982.0, 83.0])
    const4 = np.array([-6277237.0, 1585743.0, -153855.0, 6949.0])
    xmu = 4.0 * a * a
    c1 = xmu - 1.0
    c2 = constf[0] * c1 * (const2[0] + xmu * const2[1])
    c3 = constf[1] * c1 * (const3[0] + xmu * (const3[1] + xmu * const3[2]))
    c4 = (
        constf[2]
        * c1
        * (const4[0] + xmu * (const4[1] + xmu * (const4[2] + xmu * const4[3])))
    )
    r = np.arange(j + 1, n + 1)
    j0[j:] = (0.5 * a - 0.25 + r) * np.pi
    r = 0.125 / (j0[j:] ** 2)  # /
    j0[j:] = j0[j:] * (1.0 - r * (c1 + r * (c2 + r * (c3 + r * c4))))
    return j0
