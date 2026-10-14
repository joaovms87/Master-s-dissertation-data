import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt


# ---------------- PARAMETERS ----------------
beta = 0.1
J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25

h_0 = hz0
DELTA = Delta_h

n_points = np.arange(3, 14)
tau_ad_max_list = np.empty(len(n_points), complex)
tau_ad_avg_list = np.empty(len(n_points), complex)


# ---------------- PAULI MATRICES ----------------
Id = np.array([[1., 0.], [0., 1.]])
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])


# ---------------- TENSOR ----------------
def tensor(mats):
    out = mats[0]

    for m in mats[1:]:
        out = np.kron(out, m)

    return out


# ---------------- PRECOMPUTE OPERATORS ----------------
def build_operators(n):

    dim = 2**n

    S_z = np.zeros((dim, dim), dtype=np.complex128)
    S_x = np.zeros((dim, dim), dtype=np.complex128)
    S_zz = np.zeros((dim, dim), dtype=np.complex128)

    for j in range(n):
        ops = [Id] * n
        ops[j] = sigma_z
        S_z += tensor(ops)

    for j in range(n):
        ops = [Id] * n
        ops[j] = sigma_x
        S_x += hx * tensor(ops)

    for j in range(n - 1):
        ops = [Id] * n
        ops[j], ops[j + 1] = sigma_z, sigma_z
        S_zz += J * tensor(ops)

    # boundary terms
    ops = [Id] * n
    ops[0] = sigma_z
    term1 = h1 * tensor(ops)

    ops = [Id] * n
    ops[-1] = sigma_z
    termL = hL * tensor(ops)

    H_static = S_x + S_zz + term1 + termL

    return S_z, H_static


# ---------------- FIELD ----------------
def h(t, tau):

    if t == 0:
        return h_0

    return h_0 + DELTA / tau * t


def Ham(t, tau, S_z, H_static):
    return H_static + h(t, tau) * S_z


def del_H(S_z):
    return S_z


for q, L in enumerate(n_points):
    print(L)
    S_z, H_static = build_operators(L)

    E2_matrix = np.zeros_like(H_static, dtype=np.complex128)
    F_matrix = np.zeros_like(H_static, dtype=np.complex128)

    tau = 1.0
    aux = []

    t_points = np.linspace(0, tau, 10)

    for t in t_points:
        # Diagonalize Hamiltonian
        values, vectors = sl.eigh(Ham(t, tau, S_z, H_static))

        # Build E2_matrix using broadcasting
        diffs = values[:, None] - values[None, :]
        E2_matrix = diffs ** 2
        np.fill_diagonal(E2_matrix, np.inf)  # avoid i=i

        # Build F_matrix using one matrix multiplication
        F_matrix = np.abs(vectors.conj().T @ del_H(S_z) @ vectors)
        # print(F_matrix)

        # Compute X
        X = DELTA * F_matrix / E2_matrix

        aux.append(X.max())

    tau_ad_max_list[q] = max(aux)
    tau_ad_avg_list[q] = np.mean(aux)

np.savetxt('tau_ad_max.txt', tau_ad_max_list)
np.savetxt('tau_ad_avg.txt', tau_ad_avg_list)


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(n_points, np.real(tau_ad_max_list), 'k.', label=r'$\tau_{ad}$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\tau_{ad}J/\hbar$')
ax.set_box_aspect(0.85)
plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(n_points, np.real(tau_ad_avg_list), 'k.', label=r'$\tau_{ad}$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\tau_{ad}J/\hbar$')
ax.set_box_aspect(0.85)
plt.show()
