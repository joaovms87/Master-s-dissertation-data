import numpy as np
import scipy.linalg as sl
from scipy.optimize import fsolve


# Let's solve for the 3 spins Hamiltonian with one varying field

DELTA_list = [1, -1]  # total variation of gamma1
beta = 1
h2 = 1
h3 = 1


def h(t):  # protocol for gamma1 variation
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


def rho_eq(b, t):
    eigenvalues, eigenvectors = sl.eigh(Ham(t))
    Z = partition(b, eigenvalues)
    p = 0
    for i in range(8):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)
        p += np.exp(-b*eigenvalues[i])*np.outer(v, v)

    return p/Z


def rho_ad(t):
    eigenvalues, eigenvectors = sl.eigh(Ham(t))
    eigenvalues0, eigenvectors0 = sl.eigh(Ham(0))
    Z = partition(beta, eigenvalues0)
    p = 0
    for i in range(8):
        v = eigenvectors[:, i].copy()
        v /= np.linalg.norm(v)
        p += np.exp(-beta*eigenvalues0[i])*np.outer(v, v)

    return p / Z


def Ham(t):
    h1 = h(t)
    H_t = np.zeros([8, 8], complex)
    H_t[0, 0], H_t[1, 1], H_t[2, 2], H_t[3, 3], H_t[4, 4], H_t[5, 5], H_t[6, 6], H_t[7, 7] = -h1-h2-h3, h1-h2-h3, -h1+h2-h3, -h1-h2+h3, -h1+h2+h3, h1-h2+h3, h1+h2-h3, h1+h2+h3
    H_t[0, 4], H_t[0, 5], H_t[0, 6] = -1, -1, -1
    H_t[1, 2], H_t[1, 3], H_t[1, 7] = -1, -1, -1
    H_t[2, 1], H_t[2, 3], H_t[2, 7] = -1, -1, -1
    H_t[3, 1], H_t[3, 2], H_t[3, 7] = -1, -1, -1
    H_t[4, 0], H_t[4, 5], H_t[4, 6] = -1, -1, -1
    H_t[5, 0], H_t[5, 4], H_t[5, 6] = -1, -1, -1
    H_t[6, 0], H_t[6, 4], H_t[6, 5] = -1, -1, -1
    H_t[7, 1], H_t[7, 2], H_t[7, 3] = -1, -1, -1
    return H_t


def partition(b, eigenvalues):
    e1, e2, e3, e4, e5, e6, e7, e8 = eigenvalues
    return np.exp(-b*e1) + np.exp(-b*e2) + np.exp(-b*e3) + np.exp(-b*e4) + \
           np.exp(-b*e5) + np.exp(-b*e6) + np.exp(-b*e7) + np.exp(-b*e8)


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1-prod2)


def W(p):  # calculates total work done in the process
    Ef = np.trace(np.matmul(p, Ham(tau)))
    return np.real(Ef - E0)


def W_ad(t):  # calculates the work done up to time t if the evolution is adiabatic
    Ef = np.trace(np.matmul(rho_ad(t), Ham(t)))
    return np.real(Ef - E0)


def f(r, t):  # array of functions at the right-hand-side of the simultaneous equations
    p = np.zeros([8, 8], complex)
    p[0] = r[:8].copy()
    p[1, 1:] = r[8:15].copy()
    p[2, 2:] = r[15:21].copy()
    p[3, 3:] = r[21:26].copy()
    p[4, 4:] = r[26:30].copy()
    p[5, 5:] = r[30:33].copy()
    p[6, 6:] = r[33:35].copy()
    p[7, 7:] = r[35:].copy()
    for k in range(1, 8):
        for z in range(k):
            p[k,z] = np.conj(p[z,k])

    d = - 1j * (Ham(t)@p - p@Ham(t))

    new_r = np.concatenate((d[0], d[1, 1:], d[2, 2:], d[3, 3:], d[4, 4:], d[5, 5:], d[6, 6:], d[7, 7:]), axis=0)

    return new_r


# The next functions are for the adaptive Runge-Kutta method
def ratio(a, b):
    aux = (a-b)/30
    eps = np.linalg.norm(aux)
    return u*delta/eps


def new_u(u, p):
    u_new = u*p**0.25
    if u_new>2*u:
        return 2*u
    else:
        return u_new


# list of the process durations that will be investigated
taupoints = np.logspace(-1, 2, 50)
l = len(taupoints)
# lists of total work and final D - each entry corresponds to a process with tau in taupoints:
Wpoints = np.empty(l, complex)
WADpoints = np.empty(l, complex)
D2points = np.empty(l, complex)   # D[rho(tau)||rho_ad(tau)]
D_star = np.empty(l, complex)     # D[rho(tau)||pi_star(tau)]
Delta_S = np.empty(l, complex)
Delta_S_tilde = np.empty(l, complex)
beta_tilde = np.empty(l, complex)
D_tilde = np.empty(l, complex)
# compute these quantities for each tau

delta = 1e-11  # required accuracy per unit time

for q, h_0 in enumerate([1.1, 2.1]):
    DELTA = DELTA_list[q]
    # array of rho elements at time 0; necessary to apply adaptive Runge-Kutta method
    rho0 = rho_eq(beta, 0)
    for j, tau in enumerate(taupoints):
        r0 = np.concatenate((rho0[0].copy(), rho0[1, 1:].copy(), rho0[2, 2:].copy(), rho0[3, 3:].copy(), rho0[4, 4:].copy(),
                    rho0[5, 5:].copy(), rho0[6, 6:].copy(), rho0[7, 7:].copy()), axis=0)
        t = 0
        N = 100
        u = tau/N  # Initial step size
        print(tau)

        def equation_ad(b):
            return np.trace((rho_ad(tau) - rho_eq(b, tau)) @ Ham(tau))
        def equation_rho(b):
            return np.trace((rho - rho_eq(b, tau)) @ Ham(tau))
        # Initial guess
        initial_guess = beta
        # Solve the equation
        beta_star = fsolve(equation_ad, initial_guess)[0]

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = r0.copy()
        while t<tau:
            # two steps of size u
            k1 = u*f(r, t)
            k2 = u*f(r+0.5*k1, t+0.5*u)
            k3 = u*f(r+0.5*k2, t+0.5*u)
            k4 = u*f(r+k3, t+u)
            r1 = r + (k1+2*k2+2*k3+k4)/6
            k1 = u*f(r1, t+u)
            k2 = u*f(r1+0.5*k1, t+1.5*u)
            k3 = u*f(r1+0.5*k2, t+1.5*u)
            k4 = u*f(r1+k3, t+2*u)
            r1 += (k1+2*k2+2*k3+k4)/6
            # one step of size 2h
            k1 = 2*u*f(r, t)
            k2 = 2*u*f(r+0.5*k1, t+u)
            k3 = 2*u*f(r+0.5*k2, t+u)
            k4 = 2*u*f(r+k3, t+2*u)
            r2 = r + (k1+2*k2+2*k3+k4)/6

            p = ratio(r1, r2)
            if p>=1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t+2*u  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            u = new_u(u, p)  # changes h regardless of the value of p

        # build the density matrix at time tau for the process of duration tau
        rho = np.empty([8, 8], complex)
        rho[0] = r[:8].copy()
        rho[1, 1:] = r[8:15].copy()
        rho[2, 2:] = r[15:21].copy()
        rho[3, 3:] = r[21:26].copy()
        rho[4, 4:] = r[26:30].copy()
        rho[5, 5:] = r[30:33].copy()
        rho[6, 6:] = r[33:35].copy()
        rho[7, 7:] = r[35:].copy()
        for k in range(1, 8):
            for z in range(k):
                rho[k,z] = np.conj(rho[z,k])

        # initial energy for computation of the work:
        E0 = np.trace(np.matmul(rho0, Ham(0)))
        # compute the work done in the process of duration tau and save it
        Wpoints[j] = W(rho)
        # compute D[rho(tau)||rho_ad(tau)] and save it
        D2points[j] = D(rho, rho_ad(tau))
        # compute W_ad(tau) for comparison
        WADpoints[j] = W_ad(tau)

        # compute D[rho(tau)||pi_star(tau)]
        D_star[j] = D(rho, rho_eq(beta_star, tau))
        # compute Delta_S
        Delta_S[j] = - np.trace(rho_eq(beta_star, tau) @ sl.logm(rho_eq(beta_star, tau))) + np.trace(rho0 @ sl.logm(rho0))
        # compute beta_tilde and Delta S_tilde
        beta_tilde[j] = fsolve(equation_rho, initial_guess)[0]
        Delta_S_tilde[j] = - np.trace(rho_eq(beta_tilde[j], tau) @ sl.logm(rho_eq(beta_tilde[j], tau))) + np.trace(rho0 @ sl.logm(rho0))
        D_tilde[j] = D(rho_ad(tau), rho_eq(beta_tilde[j], tau))

    # save everything
    np.savetxt(f'D2 beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', D2points)
    np.savetxt(f'Wex beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', Wpoints - WADpoints)
    np.savetxt(f'beta_star beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', np.array([beta_star]))
    np.savetxt(f'beta_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', beta_tilde)
    np.savetxt(f'Delta_S beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', Delta_S)
    np.savetxt(f'Delta_S_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', Delta_S_tilde)
    np.savetxt(f'D_star beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', D_star)
    np.savetxt(f'D_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', D_tilde)

    print(beta_star*(Wpoints - WADpoints) - D_star + Delta_S)
    print()
    print(beta_tilde*(Wpoints - WADpoints) - Delta_S_tilde + D_tilde)

