import math, mpmath as mp

def poles(m, nu):
    w = math.sqrt(1+m*m+nu*nu); d = 1+m*m
    ap, bp = m*(w-nu)/d, (w-nu)/d        # beta_+ = (w-nu)(m+i)/(1+m^2)
    am, bm = -m*(w+nu)/d, (w+nu)/d       # beta_- = (w+nu)(-m+i)/(1+m^2)
    return ap, bp, am, bm

def V(t, m, nu):                          # candidate antiderivative of (nu sinh t - m)/Q
    ap, bp, am, bm = poles(m, nu); y = math.exp(t)
    return math.atan2(-bm, y-am) - math.atan2(-bp, y-ap)

def integrand(t, m, nu):
    A,B,C = 1+m*m, 2*m*nu, 1+nu*nu
    return (nu*math.sinh(t)-m)/(A*math.sinh(t)**2+B*math.sinh(t)+C)

def J_direct(m, nu, t0, t1):
    return 0.5*float(mp.quad(lambda t: t*integrand(t,m,nu), [t0,t1]))

def J_ibp(m, nu, t0, t1):                 # J = 1/2 [t V] - 1/2 int V dt
    bdry = 0.5*(t1*V(t1,m,nu) - t0*V(t0,m,nu))
    intV = 0.5*float(mp.quad(lambda t: V(t,m,nu), [t0,t1]))
    return bdry - intV

# check 1: V' == integrand (finite diff), and check 2: J_ibp == J_direct
worstVp = worstJ = 0.0
for (m,nu,X0,X1) in [(0.7,-0.4,0.3,1.5),(-1.3,0.9,-0.8,0.9),(2.5,-1.5,-2.0,-0.2),(0.05,2.0,0.1,3.0)]:
    t0,t1 = math.asinh(X0), math.asinh(X1)
    for t in [t0+0.13*(t1-t0)*k for k in range(1,7)]:
        h=1e-6; Vp=(V(t+h,m,nu)-V(t-h,m,nu))/(2*h)
        worstVp=max(worstVp, abs(Vp-integrand(t,m,nu)))
    worstJ=max(worstJ, abs(J_ibp(m,nu,t0,t1)-J_direct(m,nu,t0,t1)))
print(f"max |V'(t) - (nu sinh t - m)/Q|      = {worstVp:.1e}")
print(f"max |J_ibp(real route) - J_direct|   = {worstJ:.1e}")


# --- "no further collapse" check: beta_+ beta_- = -1 but does not fold the two Bloch-Wigner terms ---
def _betas_c(m, nu):
    import cmath
    w = math.sqrt(1+m*m+nu*nu); d = 1+m*m
    return (w-nu)*(m+1j)/d, (w+nu)*(-m+1j)/d

def _D(z):
    import cmath
    return complex(mp.polylog(2, z)).imag + cmath.phase(1-z)*math.log(abs(z))

import random
random.seed(1)
wp = wA = wB = 0.0; phisum = []
for _ in range(300):
    m, nu = random.uniform(-3,3), random.uniform(-3,3)
    y = math.exp(random.uniform(-2,2))
    bp, bm = _betas_c(m, nu)
    wp = max(wp, abs(bp*bm + 1))                 # beta_+ beta_- = -1
    wA = max(wA, abs(_D(y/bm) + _D(y/bp)))       # D(y/b-) = -D(y/b+) ?  (inversion collapse)
    wB = max(wB, abs(_D(y/bm) - _D(y/bp)))       # D(y/b-) =  D(y/b+) ?
    w = math.sqrt(1+m*m+nu*nu)
    phisum.append(math.atan2(y, w-nu-m*y) + math.atan2(y, w+nu+m*y))
print(f"collapse: max|beta_+ beta_- + 1|={wp:.1e}  |D(y/b-)+D(y/b+)|={wA:.2f}  "
      f"|D(y/b-)-D(y/b+)|={wB:.2f}  phi_++phi_- in [{min(phisum):.2f},{max(phisum):.2f}] (not const)")
