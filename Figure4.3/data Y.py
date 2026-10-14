import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from numpy.random import normal
from scipy.optimize import linear_sum_assignment
from scipy.optimize import fsolve
from scipy.stats import entropy

import matplotlib.pyplot as plt
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tau = 1.0

L = 4
n_spins = L

beta = 1.0

J_mean, J_std = 1.0, 0.1
hx_mean, hx_std = 1.1, 0.1
hz0, Delta_h = 0.1, 2.0
h1, hL = 0.25, -0.25

rng = np.random.default_rng(42)

J_list = rng.normal(J_mean, J_std, L - 1)
hx_list = rng.normal(hx_mean, hx_std, L)

h_0 = hz0
DELTA = Delta_h

Id = np.array([[1., 0.],
               [0., 1.]])
sigma_x = np.array([[0., 1.],
                   [1., 0.]])
sigma_z = np.array([[1., 0.],
                   [0., -1.]])

def tensor(matrices):
    tens = np.kron(matrices[0], matrices[1])
    for k in range(2, len(matrices)):
        tens = np.kron(tens, matrices[k])
    return tens


def h(t):  # protocol for gamma1 variation
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


def Ham(t):
    h_list = np.ones(n_spins, float)*h(t)

    s_z = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_z
        s_z += h_list[j]*tensor(matrix_list)

    s_x = np.zeros([2 ** n_spins, 2 ** n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_x
        s_x += hx_list[j] * tensor(matrix_list)

    s_z_int = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins-1):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j], matrix_list[j+1] = sigma_z, sigma_z
        s_z_int += J_list[j]*tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[0] = sigma_z
    term1 = h1 * tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[-1] = sigma_z
    termL = hL * tensor(matrix_list)

    return s_z + s_x + s_z_int + term1 + termL


def Ham_rev(t):
    h_list = np.ones(n_spins, float)*h(tau - t)

    s_z = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_z
        s_z += -h_list[j]*tensor(matrix_list)

    s_x = np.zeros([2 ** n_spins, 2 ** n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_x
        s_x += -hx_list[j] * tensor(matrix_list)

    s_z_int = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins-1):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j], matrix_list[j+1] = sigma_z, sigma_z
        s_z_int += J_list[j]*tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[0] = sigma_z
    term1 = - h1 * tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[-1] = sigma_z
    termL = - hL * tensor(matrix_list)

    return s_z + s_x + s_z_int + term1 + termL


def rho_eq_diag(b):
    eigenvals = sl.eigvalsh(Ham(tau))
    pr = np.exp(-b*eigenvals)
    pr = pr/sum(pr)
    return pr

def rho_eq(t, b):
  eigenvalues, Q = sl.eigh(Ham(t))
  aux = Q @ np.diag(np.exp(-b*eigenvalues)) @ Q.conj().T
  return aux/np.trace(aux)


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1-prod2)


def equation_line(b):
    eigs0 = sl.eigvalsh(Ham(0))
    rho_eq_diag0 = np.exp(-beta*eigs0)
    rho_eq_diag0 /= sum(rho_eq_diag0)
    return entropy(rho_eq_diag0) - entropy(rho_eq_diag(b))


def f(p, t):  # array of functions at the right-hand-side of the simultaneous equations
    return - 1j * (Ham(t) @ p - p @ Ham(t))

def f_rev(p, t):
    return - 1j * (Ham_rev(t) @ p - p @ Ham_rev(t))


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

initial_guess = beta
beta_line = np.real(fsolve(equation_line, initial_guess)[0])
print(f'beta_line = {beta_line}')

H0 = Ham(0)
Hf = Ham(tau)
vals0, vecs0 = sl.eigh(H0)
valsf, vecsf = sl.eigh(Hf)
part0 = np.sum(np.exp(-beta*vals0))

H0_tilde = Ham_rev(0)
Hf_tilde = Ham_rev(tau)
vals0_tilde, vecs0_tilde = sl.eigh(H0_tilde)
valsf_tilde, vecsf_tilde = sl.eigh(Hf_tilde)

partf_beta_line = np.sum(np.exp(-beta_line*valsf))


p_forward_points = []
p_reversed_points = []
Y_line_points = []



delta = 1e-12
for n in range(int(2**L)):
    for m in range(int(2**L)):
        print(f'n,m = {n},{m}')
        Y_line = beta_line*valsf[m] - beta*vals0[n] + np.log(partf_beta_line) - np.log(part0)
        Y_line_points.append(Y_line)

        print('FORWARD')
        v = vecs0[:, n].copy()

        r0 = np.outer(v, np.conj(v))

        t = 0
        N = 100
        u = tau/N  # Initial step size

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = r0.copy()
        while t < tau:
            # two steps of size u
            k1 = u * f(r, t)
            k2 = u * f(r + 0.5 * k1, t + 0.5 * u)
            k3 = u * f(r + 0.5 * k2, t + 0.5 * u)
            k4 = u * f(r + k3, t + u)
            r1 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            k1 = u * f(r1, t + u)
            k2 = u * f(r1 + 0.5 * k1, t + 1.5 * u)
            k3 = u * f(r1 + 0.5 * k2, t + 1.5 * u)
            k4 = u * f(r1 + k3, t + 2 * u)
            r1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # one step of size 2h
            k1 = 2 * u * f(r, t)
            k2 = 2 * u * f(r + 0.5 * k1, t + u)
            k3 = 2 * u * f(r + 0.5 * k2, t + u)
            k4 = 2 * u * f(r + k3, t + 2 * u)
            r2 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            p = ratio(r1, r2)
            if p >= 1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t + 2 * u  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            u = new_u(u, p)  # changes h regardless of the value of p

        print(f'trace = {np.trace(r)}')
        print(f'tr(P^2 - P) = {np.trace(r@r - r)}')

        v = vecsf[:, m].copy()
        Pf = np.outer(v, np.conj(v))

        p_forward = np.exp(-beta*vals0[n])/part0 * np.trace(Pf @ r)
        p_forward_points.append(p_forward)

        print()
        print()

        #####################################################################################################

        print('TIME-REVERSED')
        v = vecs0_tilde[:, m].copy()

        r0 = np.outer(v, np.conj(v))

        t = 0
        N = 100
        u = tau/N  # Initial step size

        # Solve system of differential equations by the adaptive Runge-Kutta method
        r = r0.copy()
        while t < tau:
            # two steps of size u
            k1 = u * f_rev(r, t)
            k2 = u * f_rev(r + 0.5 * k1, t + 0.5 * u)
            k3 = u * f_rev(r + 0.5 * k2, t + 0.5 * u)
            k4 = u * f_rev(r + k3, t + u)
            r1 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            k1 = u * f_rev(r1, t + u)
            k2 = u * f_rev(r1 + 0.5 * k1, t + 1.5 * u)
            k3 = u * f_rev(r1 + 0.5 * k2, t + 1.5 * u)
            k4 = u * f_rev(r1 + k3, t + 2 * u)
            r1 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # one step of size 2h
            k1 = 2 * u * f_rev(r, t)
            k2 = 2 * u * f_rev(r + 0.5 * k1, t + u)
            k3 = 2 * u * f_rev(r + 0.5 * k2, t + u)
            k4 = 2 * u * f_rev(r + k3, t + 2 * u)
            r2 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            p = ratio(r1, r2)
            if p >= 1:  # precision greater than required
                r = r1  # keep the most precise result
                t = t + 2 * u  # next t
            # if p>=1 is not satisfied, repeat everything without going to next t
            u = new_u(u, p)  # changes h regardless of the value of p

        print(f'trace = {np.trace(r)}')
        print(f'tr(P^2 - P) = {np.trace(r@r - r)}')

        v = vecsf_tilde[:, n].copy()
        Pf = np.outer(v, np.conj(v))

        p_reversed = np.exp(-beta_line*valsf[m])/partf_beta_line * np.trace(Pf @ r)
        p_reversed_points.append(p_reversed)



p_forward_points = np.array(p_forward_points)
p_reversed_points = np.array(p_reversed_points)

ln_ratio = np.log(p_forward_points/p_reversed_points)

np.savetxt(f'ln_ratios Y {L} spins, beta={beta}, h0={h_0}. DELTA={DELTA}', ln_ratio)
np.savetxt(f'Y for {L} spins, beta={beta}, h0={h_0}. DELTA={DELTA}', np.array(Y_line_points))

"""
plt.plot(X_points, ln_ratio)
plt.show()
"""
