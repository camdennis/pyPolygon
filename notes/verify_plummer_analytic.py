"""
Verification of the fully analytic (closed-form) double-boundary integrals for the
softened-log (Plummer) overlap, its shape gradient, and the Plummer self-repulsion.

Conventions:
  - polygons CCW; edge tangent t = e/|e|; outward normal n = (t_y, -t_x)
  - kernel  K_sigma(r) = sigma^2 / (pi (r^2+sigma^2)^2)
  - field   F(r) = r / (2 pi (r^2 + sigma^2)),  potential Phi = ln(r^2+sigma^2)/(4 pi)
  - Psi_B(y) = oint_{dB} F(z-y).n_B dz   (softened indicator / winding of B)
  - A_cap^sigma = int_A Psi_B dy = -(1/4pi) sum_{a,b} (nA.nB) LA LB I_ab,
        I_ab = int_0^1 int_0^1 ln Q ds dt,  Q = |y(s)-z(t)|^2 + sigma^2

Edge-pair coordinates (edge a of A: y = p0 + s avec; edge b of B: z = q0 + t bvec):
  P0 = (p0-q0).bhat, P1 = avec.bhat, X0 = cross(p0-q0, bhat), X1 = cross(avec, bhat)
  p(s) = P0 + P1 s ; chi(s) = X0 + X1 s ; h^2 = chi^2 + sigma^2
  u_e(s) = p(s) - e LB  (e = 0,1) ;  Q(s,t) = (p - LB t)^2 + chi^2 + sigma^2
  after xi = chi(s):  u_e = alpha xi + beta_e,  alpha = P1/X1, beta_e = U0e - alpha X0
  Q2(xi) = (1+alpha^2) xi^2 + 2 alpha beta xi + beta^2 + sigma^2 = |y - z_e|^2 + sigma^2

Companion to plummerOverlap.tex: every numbered closed form there is implemented
and verified here against adaptive (tanh-sinh) quadrature at 25-digit precision.
"""
import mpmath as mp
mp.mp.dps = 25

def cross(u, v): return u[0]*v[1] - u[1]*v[0]
def dot(u, v):   return u[0]*v[0] + u[1]*v[1]
def sub(u, v):   return (u[0]-v[0], u[1]-v[1])
def norm(u):     return mp.sqrt(dot(u, u))

PI = mp.pi

# ---------------------------------------------------------------------------
# 0. softened winding number Psi_B: closed form vs 2D convolution quadrature
# ---------------------------------------------------------------------------
def Psi_closed(y, poly, sigma):
    n = len(poly); s = mp.mpf(0)
    for i in range(n):
        v0, v1 = poly[i], poly[(i+1) % n]
        e = sub(v1, v0); L = norm(e); that = (e[0]/L, e[1]/L)
        nhat = (that[1], -that[0])                     # outward for CCW
        w = sub(v0, y)
        wpar = dot(w, that); wperp = dot(w, nhat)
        h = mp.sqrt(wperp**2 + sigma**2)
        s += wperp/h * (mp.atan((wpar+L)/h) - mp.atan(wpar/h))
    return s/(2*PI)

def K_sigma(r2, sigma): return sigma**2/(PI*(r2+sigma**2)**2)

def tri_quad(f, A, B, C):
    """integrate f over triangle ABC (2D tanh-sinh, nested)"""
    J = abs(cross(sub(B, A), sub(C, A)))
    def inner(u):
        return mp.quad(lambda v: f((A[0]+u*(B[0]-A[0])+v*(C[0]-A[0]),
                                    A[1]+u*(B[1]-A[1])+v*(C[1]-A[1]))),
                       [0, 1-u])
    return J*mp.quad(inner, [0, 1])

def poly_quad(f, poly):
    c = (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))
    tot = mp.mpf(0)
    for i in range(len(poly)):
        tot += tri_quad(f, c, poly[i], poly[(i+1) % len(poly)])
    return tot

def test_Psi():
    poly = [(0, 0), (1.3, -0.2), (1.7, 0.9), (0.6, 1.4), (-0.3, 0.8)]
    poly = [(mp.mpf(a), mp.mpf(b)) for a, b in poly]
    sigma = mp.mpf('0.17')
    worst = 0
    for y in [(0.5, 0.5), (2.5, 1.5), (0.9, -0.05), (-1.0, 3.0)]:
        y = (mp.mpf(y[0]), mp.mpf(y[1]))
        direct = poly_quad(lambda z: K_sigma((z[0]-y[0])**2+(z[1]-y[1])**2, sigma), poly)
        closed = Psi_closed(y, poly, sigma)
        worst = max(worst, abs(direct-closed))
    print("T0  Psi closed vs 2D conv quad     max err:", mp.nstr(worst, 3))

# ---------------------------------------------------------------------------
# 1. primitives
# ---------------------------------------------------------------------------
def Lam0(s, a, b, c):
    """int ln(a s^2 + b s + c) ds ; requires D=4ac-b^2>0"""
    q = a*s*s + b*s + c; D = 4*a*c - b*b; sD = mp.sqrt(D)
    return (s + b/(2*a))*mp.log(q) - 2*s + (sD/a)*mp.atan((2*a*s+b)/sD)

def Lam1(s, a, b, c):
    """int s ln(a s^2 + b s + c) ds"""
    q = a*s*s + b*s + c; D = 4*a*c - b*b; sD = mp.sqrt(D)
    return ((s*s)/2 - (b*b - 2*a*c)/(4*a*a))*mp.log(q) - s*s/2 + b*s/(2*a) \
           - (b*sD/(2*a*a))*mp.atan((2*a*s+b)/sD)

def Xi0(xi, alpha, beta, sigma):
    A = 1 + alpha**2; Delta = mp.sqrt(beta**2 + A*sigma**2)
    return mp.atan((A*xi + alpha*beta)/Delta)/Delta

def Q2(xi, alpha, beta, sigma):
    return (1+alpha**2)*xi*xi + 2*alpha*beta*xi + beta*beta + sigma*sigma

def Rprim(xi, alpha, beta, sigma):
    """int xi (alpha sigma^2 - beta xi)/Q2 dxi"""
    A = 1 + alpha**2
    x0 = Xi0(xi, alpha, beta, sigma)
    x1 = mp.log(Q2(xi, alpha, beta, sigma))/(2*A) - (alpha*beta/A)*x0
    return -beta*xi/A + (alpha*(sigma**2*A + 2*beta**2)/A)*x1 \
           + (beta*(beta**2+sigma**2)/A)*x0

def Theta(xi, alpha, beta, sigma):
    return mp.atan((alpha*xi+beta)/mp.sqrt(xi*xi+sigma*sigma))

def M1(xi, alpha, beta, sigma):
    """int  xi/sqrt(xi^2+s^2) * Theta  dxi   (elementary)"""
    A = 1 + alpha**2; Delta = mp.sqrt(beta**2 + A*sigma**2)
    return mp.sqrt(xi*xi+sigma*sigma)*Theta(xi, alpha, beta, sigma) \
           + (beta/(2*A))*mp.log(Q2(xi, alpha, beta, sigma)) \
           - (alpha*Delta/A)*mp.atan((A*xi+alpha*beta)/Delta)

def Tprim(xi, alpha, beta, sigma):
    """int asinh(xi/sigma) (alpha sigma^2 - beta xi)/(sqrt(xi^2+sigma^2) Q2) dxi
       via eta = xi + sqrt(xi^2+sigma^2):  = 2 sum_k c_k [ln(eta/sigma) ln(1-eta/eta_k)
                                                          + Li2(eta/eta_k)]"""
    A = 1 + alpha**2
    Delta = mp.sqrt(beta**2 + A*sigma**2)
    zetas = [(-alpha*beta + 1j*Delta)/A, (-alpha*beta - 1j*Delta)/A]
    etas = []
    for z in zetas:
        r = mp.sqrt(z*z + sigma**2)
        etas += [z + r, z - r]
    def Qq_p(e):   # derivative of quartic Qq(eta)=4 eta^2 Q2(xi(eta))
        return 4*e*A*(e*e-sigma**2) + 4*alpha*beta*(3*e*e-sigma**2) + 8*(beta**2+sigma**2)*e
    def N(e):
        return -beta*e*e + 2*alpha*sigma**2*e + beta*sigma**2
    eta = xi + mp.sqrt(xi*xi + sigma*sigma)
    tot = mp.mpc(0)
    for ek in etas:
        ck = N(ek)/Qq_p(ek)
        tot += ck*(mp.log(eta/sigma)*mp.log(1 - eta/ek) + mp.polylog(2, eta/ek))
    return 2*mp.re(tot)

def Vplus(xi, sigma):  return (xi*mp.sqrt(xi*xi+sigma*sigma) + sigma**2*mp.asinh(xi/sigma))/2
def Vminus(xi, sigma): return (xi*mp.sqrt(xi*xi+sigma*sigma) - sigma**2*mp.asinh(xi/sigma))/2

def M2(xi, alpha, beta, sigma):
    """int sqrt(xi^2+s^2) Theta dxi = V+ Theta - R/2 - (s^2/2) T"""
    return Vplus(xi, sigma)*Theta(xi, alpha, beta, sigma) \
           - Rprim(xi, alpha, beta, sigma)/2 - (sigma**2/2)*Tprim(xi, alpha, beta, sigma)

def M1p(xi, alpha, beta, sigma):
    """int xi^2/sqrt(xi^2+s^2) Theta dxi = V- Theta - R/2 + (s^2/2) T"""
    return Vminus(xi, sigma)*Theta(xi, alpha, beta, sigma) \
           - Rprim(xi, alpha, beta, sigma)/2 + (sigma**2/2)*Tprim(xi, alpha, beta, sigma)

def test_primitives():
    import random; random.seed(3)
    worst = {k: mp.mpf(0) for k in ("Lam0", "Lam1", "R", "M1", "T", "M2", "M1p")}
    for _ in range(6):
        a = mp.mpf(random.uniform(0.3, 3)); b = mp.mpf(random.uniform(-2, 2))
        c = (b*b/(4*a)) + mp.mpf(random.uniform(0.05, 2))
        s0, s1 = mp.mpf(random.uniform(-1.5, 0)), mp.mpf(random.uniform(0.2, 1.8))
        worst["Lam0"] = max(worst["Lam0"], abs(Lam0(s1,a,b,c)-Lam0(s0,a,b,c) -
                            mp.quad(lambda s: mp.log(a*s*s+b*s+c), [s0, s1])))
        worst["Lam1"] = max(worst["Lam1"], abs(Lam1(s1,a,b,c)-Lam1(s0,a,b,c) -
                            mp.quad(lambda s: s*mp.log(a*s*s+b*s+c), [s0, s1])))
        al = mp.mpf(random.uniform(-3, 3)); be = mp.mpf(random.uniform(-2, 2))
        sg = mp.mpf(random.uniform(0.05, 0.6))
        x0, x1 = mp.mpf(random.uniform(-1.5, -0.1)), mp.mpf(random.uniform(0.1, 1.5))
        worst["R"] = max(worst["R"], abs(Rprim(x1,al,be,sg)-Rprim(x0,al,be,sg) -
                        mp.quad(lambda x: x*(al*sg**2-be*x)/Q2(x,al,be,sg), [x0, x1])))
        worst["M1"] = max(worst["M1"], abs(M1(x1,al,be,sg)-M1(x0,al,be,sg) -
                        mp.quad(lambda x: x/mp.sqrt(x*x+sg*sg)*Theta(x,al,be,sg), [x0, x1])))
        worst["T"] = max(worst["T"], abs(Tprim(x1,al,be,sg)-Tprim(x0,al,be,sg) -
                        mp.quad(lambda x: mp.asinh(x/sg)*(al*sg**2-be*x) /
                                (mp.sqrt(x*x+sg*sg)*Q2(x,al,be,sg)), [x0, 0, x1])))
        worst["M2"] = max(worst["M2"], abs(M2(x1,al,be,sg)-M2(x0,al,be,sg) -
                        mp.quad(lambda x: mp.sqrt(x*x+sg*sg)*Theta(x,al,be,sg), [x0, x1])))
        worst["M1p"] = max(worst["M1p"], abs(M1p(x1,al,be,sg)-M1p(x0,al,be,sg) -
                        mp.quad(lambda x: x*x/mp.sqrt(x*x+sg*sg)*Theta(x,al,be,sg), [x0, x1])))
    for k, v in worst.items():
        print(f"T1  primitive {k:5s} vs quadrature   max err:", mp.nstr(v, 3))

# ---------------------------------------------------------------------------
# 2. edge-pair coordinates, closed I = int int ln Q, and gradient moments W0,W1
# ---------------------------------------------------------------------------
def pair_coords(p0, avec, q0, bvec, sigma):
    LA, LB = norm(avec), norm(bvec)
    bhat = (bvec[0]/LB, bvec[1]/LB)
    w0 = sub(p0, q0)
    P0, P1 = dot(w0, bhat), dot(avec, bhat)
    X0, X1 = cross(w0, bhat), cross(avec, bhat)
    return LA, LB, P0, P1, X0, X1

def Qfun(s, t, p0, avec, q0, bvec, sigma):
    y = (p0[0]+s*avec[0], p0[1]+s*avec[1]); z = (q0[0]+t*bvec[0], q0[1]+t*bvec[1])
    return (y[0]-z[0])**2 + (y[1]-z[1])**2 + sigma**2

def I_closed(p0, avec, q0, bvec, sigma, tol=mp.mpf('1e-14')):
    LA, LB, P0, P1, X0, X1 = pair_coords(p0, avec, q0, bvec, sigma)
    tot = mp.mpf(0)
    for e, eps in ((0, 1), (1, -1)):
        U0 = P0 - e*LB
        aq = LA*LA; bq = 2*(U0*P1 + X0*X1); cq = U0*U0 + X0*X0 + sigma**2
        Ae = U0*(Lam0(1,aq,bq,cq)-Lam0(0,aq,bq,cq)) + P1*(Lam1(1,aq,bq,cq)-Lam1(0,aq,bq,cq))
        Be = U0 + P1/2
        if abs(X1) > tol*LA:
            al = P1/X1; be = U0 - al*X0
            Ce = (M2(X0+X1, al, be, sigma) - M2(X0, al, be, sigma))/X1
        else:
            h = mp.sqrt(X0*X0 + sigma**2)
            Fa = lambda u: u*mp.atan(u/h) - (h/2)*mp.log(u*u+h*h)
            Ce = (h/P1)*(Fa(U0+P1) - Fa(U0))
        tot += eps*(Ae - 2*Be + 2*Ce)
    return tot/LB

def psi_edge(s, p0, avec, q0, bvec, sigma):
    """contribution of edge (q0,bvec) of B to Psi_B at y(s) on edge (p0,avec)"""
    LA, LB, P0, P1, X0, X1 = pair_coords(p0, avec, q0, bvec, sigma)
    chi = X0 + X1*s; p = P0 + P1*s; h = mp.sqrt(chi*chi + sigma**2)
    return -(chi/(2*PI*h))*(mp.atan(p/h) - mp.atan((p-LB)/h))

def W_closed(p0, avec, q0, bvec, sigma, tol=mp.mpf('1e-14')):
    """W_m = int_0^1 s^m psi_edge ds, m = 0,1 (closed form)"""
    LA, LB, P0, P1, X0, X1 = pair_coords(p0, avec, q0, bvec, sigma)
    W0 = mp.mpf(0); W1 = mp.mpf(0)
    for e, eps in ((0, 1), (1, -1)):
        U0 = P0 - e*LB
        if abs(X1) > tol*LA:
            al = P1/X1; be = U0 - al*X0
            xi0, xi1 = X0, X0 + X1
            dM1  = M1(xi1, al, be, sigma) - M1(xi0, al, be, sigma)
            dM1p = M1p(xi1, al, be, sigma) - M1p(xi0, al, be, sigma)
            W0 += -eps/(2*PI) * dM1/X1
            W1 += -eps/(2*PI) * (dM1p - X0*dM1)/(X1*X1)
        else:
            h = mp.sqrt(X0*X0 + sigma**2)
            Fa  = lambda u: u*mp.atan(u/h) - (h/2)*mp.log(u*u+h*h)     # int atan(u/h) du
            Fua = lambda u: ((u*u+h*h)/2)*mp.atan(u/h) - h*u/2          # int u atan(u/h) du
            d0 = (Fa(U0+P1) - Fa(U0))/P1
            d1 = ((Fua(U0+P1)-Fua(U0))/P1 - U0*(Fa(U0+P1)-Fa(U0))/P1)/P1
            W0 += -eps*(X0/(2*PI*h))*d0
            W1 += -eps*(X0/(2*PI*h))*d1
    return W0, W1

def rand_edges(seed):
    import random; random.seed(seed)
    r = lambda lo, hi: mp.mpf(random.uniform(lo, hi))
    return ((r(-1,1), r(-1,1)), (r(-1.5,1.5), r(-1.5,1.5)),
            (r(-1,1), r(-1,1)), (r(-1.5,1.5), r(-1.5,1.5)))

def test_pair():
    worstI = mp.mpf(0); worstW = mp.mpf(0)
    for seed in range(8):
        p0, avec, q0, bvec = rand_edges(seed)
        for sigma in (mp.mpf('0.3'), mp.mpf('0.07')):
            Iq = mp.quad(lambda s: mp.quad(lambda t:
                    mp.log(Qfun(s, t, p0, avec, q0, bvec, sigma)), [0, 1]), [0, 1])
            worstI = max(worstI, abs(Iq - I_closed(p0, avec, q0, bvec, sigma)))
            W0q = mp.quad(lambda s: psi_edge(s, p0, avec, q0, bvec, sigma), [0, 1])
            W1q = mp.quad(lambda s: s*psi_edge(s, p0, avec, q0, bvec, sigma), [0, 1])
            W0c, W1c = W_closed(p0, avec, q0, bvec, sigma)
            worstW = max(worstW, abs(W0q-W0c), abs(W1q-W1c))
    # exactly parallel pair
    p0, avec, q0, bvec = (0, 0), (1, mp.mpf('0.5')), (mp.mpf('0.2'), 1), (2, 1)
    sigma = mp.mpf('0.15')
    Iq = mp.quad(lambda s: mp.quad(lambda t:
            mp.log(Qfun(s, t, p0, avec, q0, bvec, sigma)), [0, 1]), [0, 1])
    worstI = max(worstI, abs(Iq - I_closed(p0, avec, q0, bvec, sigma)))
    W0q = mp.quad(lambda s: psi_edge(s, p0, avec, q0, bvec, sigma), [0, 1])
    W1q = mp.quad(lambda s: s*psi_edge(s, p0, avec, q0, bvec, sigma), [0, 1])
    W0c, W1c = W_closed(p0, avec, q0, bvec, sigma)
    worstW = max(worstW, abs(W0q-W0c), abs(W1q-W1c))
    print("T2  pair energy I closed vs 2D quad max err:", mp.nstr(worstI, 3))
    print("T3  gradient moments W0,W1          max err:", mp.nstr(worstW, 3))

# ---------------------------------------------------------------------------
# 3. sharp warm-up: int int ln|y-z| ds dt by complex corner formula
# ---------------------------------------------------------------------------
def test_sharp():
    p0, avec = (mp.mpf(0), mp.mpf(0)), (mp.mpf(1), mp.mpf('0.3'))
    q0, bvec = (mp.mpf('1.8'), mp.mpf('1.1')), (mp.mpf('-0.4'), mp.mpf('0.9'))
    Yc = lambda s: (p0[0]+s*avec[0]) + 1j*(p0[1]+s*avec[1])
    Zc = lambda t: (q0[0]+t*bvec[0]) + 1j*(q0[1]+t*bvec[1])
    ac, bc = avec[0]+1j*avec[1], bvec[0]+1j*bvec[1]
    H = lambda w: w*w*(2*mp.log(w) - 3)/4
    tot = mp.mpc(0)
    for i in (0, 1):
        for j in (0, 1):
            tot += (-1)**(i+j) * H(Yc(i) - Zc(j))
    closed = mp.re(-tot/(ac*bc))
    quad = mp.quad(lambda s: mp.quad(lambda t: mp.log(abs(Yc(s)-Zc(t))), [0, 1]), [0, 1])
    print("T4  sharp complex-corner formula     err:", mp.nstr(abs(closed-quad), 3))

# ---------------------------------------------------------------------------
# 4. end-to-end: two polygons, total energy + full vertex gradient vs FD
# ---------------------------------------------------------------------------
def edges(poly):
    n = len(poly)
    for i in range(n):
        v0, v1 = poly[i], poly[(i+1) % n]
        yield i, v0, sub(v1, v0)

def nhat_of(e):
    L = norm(e); return (e[1]/L, -e[0]/L), L

def A_sigma(polyA, polyB, sigma):
    tot = mp.mpf(0)
    for _, p0, av in edges(polyA):
        nA, LA = nhat_of(av)
        for _, q0, bv in edges(polyB):
            nB, LB = nhat_of(bv)
            tot += dot(nA, nB)*LA*LB*I_closed(p0, av, q0, bv, sigma)
    return -tot/(4*PI)

def grad_A_sigma(polyA, polyB, sigma):
    """d A_sigma / d v  for every vertex of A and of B (closed form)"""
    gA = [[mp.mpf(0), mp.mpf(0)] for _ in polyA]
    gB = [[mp.mpf(0), mp.mpf(0)] for _ in polyB]
    def accumulate(P, Q, g):        # boundary of P carries the hat functions
        n = len(P)
        for i, p0, av in edges(P):
            nP, LP = nhat_of(av)
            W0 = mp.mpf(0); W1 = mp.mpf(0)
            for _, q0, bv in edges(Q):
                w0, w1 = W_closed(p0, av, q0, bv, sigma)
                W0 += w0; W1 += w1
            for k in range(2):
                g[i][k]         += LP*nP[k]*(W0 - W1)     # weight (1-s) on v_i
                g[(i+1) % n][k] += LP*nP[k]*W1            # weight  s   on v_{i+1}
    accumulate(polyA, polyB, gA)
    accumulate(polyB, polyA, gB)
    return gA, gB

def test_end_to_end():
    polyA = [(0, 0), (1.2, 0.1), (1.0, 1.1), (-0.2, 0.9)]
    polyB = [(0.7, 0.5), (1.9, 0.4), (2.1, 1.6), (0.9, 1.7)]
    polyA = [(mp.mpf(a), mp.mpf(b)) for a, b in polyA]
    polyB = [(mp.mpf(a), mp.mpf(b)) for a, b in polyB]
    sigma = mp.mpf('0.12')
    Ecl = A_sigma(polyA, polyB, sigma)
    Eq = poly_quad(lambda y: Psi_closed(y, polyB, sigma), polyA)
    print("T5  total A^sigma closed vs area quad err:", mp.nstr(abs(Ecl-Eq), 3))
    gA, gB = grad_A_sigma(polyA, polyB, sigma)
    d = mp.mpf('1e-6'); worst = mp.mpf(0)
    for poly, g in ((polyA, gA), (polyB, gB)):
        for i in range(len(poly)):
            for k in range(2):
                pp = [list(v) for v in poly]; pm = [list(v) for v in poly]
                pp[i][k] += d; pm[i][k] -= d
                if poly is polyA:
                    fd = (A_sigma(pp, polyB, sigma) - A_sigma(pm, polyB, sigma))/(2*d)
                else:
                    fd = (A_sigma(polyA, pp, sigma) - A_sigma(polyA, pm, sigma))/(2*d)
                worst = max(worst, abs(fd - g[i][k]))
    print("T6  full vertex gradient vs FD        err:", mp.nstr(worst, 3))

# ---------------------------------------------------------------------------
# 5. self-repulsion: S = int int ds dt / Q^2  (fully elementary)
# ---------------------------------------------------------------------------
def S_closed(p0, avec, q0, bvec, sigma, tol=mp.mpf('1e-14')):
    LA, LB, P0, P1, X0, X1 = pair_coords(p0, avec, q0, bvec, sigma)
    tot = mp.mpf(0)
    if abs(X1) > tol*LA:
        xi0, xi1 = X0, X0 + X1
        for e, eps in ((0, 1), (1, -1)):
            U0 = P0 - e*LB
            al = P1/X1; be = U0 - al*X0
            A = 1+al**2; Delta = mp.sqrt(be**2 + A*sigma**2)
            zet = [(-al*be + 1j*Delta)/A, (-al*be - 1j*Delta)/A]
            poles = [1j*sigma, -1j*sigma] + zet
            def den_p(x):
                return 2*x*Q2(x, al, be, sigma) + (x*x+sigma**2)*(2*A*x + 2*al*be)
            def rational_int(numf):
                acc = mp.mpc(0)
                for pk in poles:
                    acc += numf(pk)/den_p(pk)*(mp.log(xi1-pk) - mp.log(xi0-pk))
                return mp.re(acc)
            term1 = rational_int(lambda x: al*x + be)            # int u_e/(h^2 q_e)
            bnd = lambda x: x*Theta(x, al, be, sigma)/(sigma**2*mp.sqrt(x*x+sigma**2))
            Mm3 = bnd(xi1) - bnd(xi0) \
                  - rational_int(lambda x: x*(al*sigma**2 - be*x))/sigma**2
            tot += eps*(term1 + Mm3)
        return tot/(2*LB*X1)
    else:
        h2 = X0*X0 + sigma**2; h = mp.sqrt(h2)
        for e, eps in ((0, 1), (1, -1)):
            U0 = P0 - e*LB
            # int_0^1 [ u/(h^2 (u^2+h^2)) + atan(u/h)/h^3 ] ds, u = U0+P1 s
            F = lambda u: (mp.log(u*u+h2)/(2*h2) + (u*mp.atan(u/h) - (h/2)*mp.log(u*u+h2))/h**3)/P1
            tot += eps*(F(U0+P1) - F(U0))
        return tot/(2*LB)

def test_selfrep():
    worst = mp.mpf(0)
    for seed in (11, 12, 13):
        p0, avec, q0, bvec = rand_edges(seed)
        for sigma in (mp.mpf('0.25'), mp.mpf('0.08')):
            Sq = mp.quad(lambda s: mp.quad(lambda t:
                    1/Qfun(s, t, p0, avec, q0, bvec, sigma)**2, [0, 1]), [0, 1])
            worst = max(worst, abs(Sq - S_closed(p0, avec, q0, bvec, sigma)))
    print("T7  self-repulsion S closed vs quad   err:", mp.nstr(worst, 3))

if __name__ == "__main__":
    test_Psi()
    test_primitives()
    test_pair()
    test_sharp()
    test_end_to_end()
    test_selfrep()
