import math, mpmath as mp
# Unified master  M[V] = int V' Theta dX = [V Theta] - int V Theta' dX,
# Theta = arctan((mX+nu)/sqrt(X^2+1)),  Theta' = (m-nuX)/((A X^2+B X+C) sqrt(X^2+1)).
# Weights:  sqrt(X^2+1) -> V=vp (energy);  X/sqrt -> V=sqrt (W0);  X^2/sqrt -> V=vm (W1).
# Claim: transcendental content is  +J (energy), 0 (W0), -J (W1),  J = J_arcsinh(m,nu).
def run(m, nu, X0, X1):
    A,B,C = 1+m*m, 2*m*nu, 1+nu*nu
    sq = lambda X: math.sqrt(X*X+1)
    Th = lambda X: math.atan2(m*X+nu, sq(X))
    vp = lambda X: 0.5*(X*sq(X)+math.asinh(X))
    vm = lambda X: 0.5*(X*sq(X)-math.asinh(X))
    q  = lambda f,a,b: float(mp.quad(f,[a,b]))
    # the three master integrals (direct)
    Me = q(lambda X: sq(X)*Th(X), X0, X1)              # energy  int sqrt*Theta
    M0 = q(lambda X: X/sq(X)*Th(X), X0, X1)            # W0      int (X/sqrt)*Theta
    M1 = q(lambda X: X*X/sq(X)*Th(X), X0, X1)          # W1      int (X^2/sqrt)*Theta
    # elementary pieces
    rho = q(lambda X: X*(m-nu*X)/(A*X*X+B*X+C), X0, X1)          # int X(m-nuX)/(AX^2+BX+C)
    tau = q(lambda X: (m-nu*X)/(A*X*X+B*X+C), X0, X1)            # int (m-nuX)/(AX^2+BX+C)
    bvp = vp(X1)*Th(X1)-vp(X0)*Th(X0)
    bvm = vm(X1)*Th(X1)-vm(X0)*Th(X0)
    bsq = sq(X1)*Th(X1)-sq(X0)*Th(X0)
    J   = 0.5*q(lambda X: math.asinh(X)*(nu*X-m)/((A*X*X+B*X+C)*sq(X)), X0, X1)  # J_arcsinh
    # claims:
    e_energy = (Me - bvp + 0.5*rho) - J          # should be 0
    e_W0     = (M0 - bsq + tau)                  # should be 0 (no J)
    e_W1     = (M1 - bvm + 0.5*rho) + J           # should be 0
    return e_energy, e_W0, e_W1, J
worst = [0,0,0]
for (m,nu,X0,X1) in [(0.7,-0.4,0.3,1.5),(-1.3,0.9,-0.8,0.9),(2.5,-1.5,-2.0,-0.2),(0.05,2.0,0.1,3.0),(-0.5,-0.5,0.5,2.5)]:
    ee,e0,e1,J = run(m,nu,X0,X1)
    worst = [max(worst[0],abs(ee)),max(worst[1],abs(e0)),max(worst[2],abs(e1))]
    print(f"m={m:+.2f} nu={nu:+.2f}: energy-J={ee:+.1e}  W0(noJ)={e0:+.1e}  W1+J={e1:+.1e}   (J={J:+.5f})")
print(f"\nworst: energy={worst[0]:.1e}  W0={worst[1]:.1e}  W1={worst[2]:.1e}")
