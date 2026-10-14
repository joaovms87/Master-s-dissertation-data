import numpy as np
import scipy.linalg as sl
from scipy.optimize import fsolve
from scipy.stats import entropy
from scipy.integrate import quad
from joblib import Parallel, delayed
import os


# ---------------- PARAMETERS ----------------
beta = 1.0
J_mean, J_std = 1.0, 0.1
hx_mean, hx_std = 1.1, 0.1
hz0, Delta_h = 0.1, 2.0
h1, hL = 0.25, -0.25

rng = np.random.default_rng(42)

L = 12

J_list = rng.normal(J_mean, J_std, L - 1)
hx_list = rng.normal(hx_mean, hx_std, L)

h_0 = hz0
DELTA = Delta_h

n_points = [12]
taupoints = np.logspace(-1, 2, 20)

# Number of parallel tau calculations
N_JOBS = 3


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
        S_x += hx_list[j] * tensor(ops)

    for j in range(n - 1):
        ops = [Id] * n
        ops[j], ops[j + 1] = sigma_z, sigma_z
        S_zz += J_list[j] * tensor(ops)

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


def h_back(t, tau):

    return h_0 + DELTA - DELTA / tau * t


# ---------------- HAMILTONIANS ----------------
def Ham(t, tau, S_z, H_static):

    return H_static + h(t, tau) * S_z


def Ham_back(t, tau, S_z, H_static):

    return H_static + h_back(t, tau) * S_z


# ---------------- EQUILIBRIUM ----------------
def rho_eq(H, beta):

    evals, evecs = sl.eigh(H)

    rho = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T

    return rho / np.trace(rho)


def rho_eq_diag(beta, evals):

    p = np.exp(-beta * evals)

    return p / np.sum(p)


def rho_eq_tau(beta, evals, evecs):

    rho = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T

    return rho / np.trace(rho)


# ---------------- EVOLUTION ----------------
def evolve(rho0, S_z, H_static, tau):

    r = rho0.copy()

    t = 0.0

    if tau < 10:
        u = tau / 100
    else:
        u = 0.03

    def f(r, t):

        H = Ham(t, tau, S_z, H_static)

        return -1j * (H @ r - r @ H)

    while t < tau:

        k1 = u * f(r, t)
        k2 = u * f(r + 0.5 * k1, t + 0.5 * u)
        k3 = u * f(r + 0.5 * k2, t + 0.5 * u)
        k4 = u * f(r + k3, t + u)

        r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t += u

    return r




# ---------------- SINGLE TAU COMPUTATION ----------------
def compute_tau(j, tau, rho0, S_z, H_static, n):

    print(f"Starting tau = {tau:.3e}")

    rho = evolve(rho0, S_z, H_static, tau)

    Hf = Ham(tau, tau, S_z, H_static)
    H0 = Ham(0, tau, S_z, H_static)

    evals, evecs = sl.eigh(Hf)

    eigvals0, eigvecs0 = sl.eigh(H0)

    # Adiabatic state
    prbs = np.exp(-beta * eigvals0)
    prbs /= prbs.sum()

    r_ad = evecs @ np.diag(prbs) @ evecs.conj().T

    ln_rho_ad = evecs @ np.diag(np.log(prbs)) @ evecs.conj().T

    E0 = np.trace(H0 @ rho0)

    Wad = np.dot(prbs, evals) - E0

    W = np.trace(Hf @ rho) - E0

    # ---------------- ADIABATIC BACKWARD EVOLUTION ----------------
    dim = 2 ** n

    # arbitrary phases
    # (they do not affect Wback or Wc)
    phases = np.ones(dim, dtype=np.complex128)

    # vectorized construction of Uad_back
    Uad_back = (
            eigvecs0
            @ (phases[:, None] * evecs.conj().T)
    )

    # backward-evolved state
    rho_back = (
            Uad_back
            @ rho
            @ Uad_back.conj().T
    )

    # backward work
    Wback = (
            np.trace(H0 @ rho_back)
            - np.trace(Hf @ rho)
    )

    # cyclic work
    Wc = W + Wback

    # ---------------- TEMPERATURES ----------------
    beta_tilde_val = fsolve(
        lambda b: np.trace(
            (rho - rho_eq_tau(b, evals, evecs)) @ Hf
        ),
        beta
    )[0]

    beta_line_val = np.real(
        fsolve(
            lambda b:
            entropy(rho_eq_diag(beta, eigvals0))
            - entropy(rho_eq_diag(b, evals)),
            beta
        )[0]
    )

    beta_star_val = np.real(
        fsolve(
            lambda b: np.trace(
                (r_ad - rho_eq_tau(b, evals, evecs)) @ Hf
            ),
            beta
        )[0]
    )

    exps_tilde = np.exp(-beta_tilde_val * evals)

    Z_tilde = np.sum(exps_tilde)

    # ---------------- DIVERGENCES ----------------
    D_ad = (
        -entropy(rho_eq_diag(beta, eigvals0))
        - np.trace(rho @ ln_rho_ad)
    )

    D_tilde_val = (
        -entropy(rho_eq_diag(beta, eigvals0))
        + beta_tilde_val * np.trace(rho @ Hf)
        + np.log(Z_tilde)
    )

    D_star_val = entropy(
        prbs,
        rho_eq_diag(beta_star_val, evals)
    )

    Wise = (
        np.sum(
            rho_eq_diag(beta_line_val, evals)
            * evals
        )
        - E0
    )

    # ---------------- OBSERVABLE ----------------
    sz_avg = np.trace(rho @ S_z)

    # ---------------- ENTROPY PRODUCTION ----------------
    w = 2**n

    m_vals = np.arange(1, w + 1)

    log_diag = np.diag(np.log(m_vals))

    log_OMEGA_f = (
        evecs @ log_diag @ evecs.conj().T
    )

    log_OMEGA_0 = (
        eigvecs0 @ log_diag @ eigvecs0.conj().T
    )

    Delta_Svol = np.trace(
        rho @ log_OMEGA_f
        - rho0 @ log_OMEGA_0
    )

    print(f"Finished tau = {tau:.3e}")

    return (
        j,
        np.real(D_ad),
        np.real(W),
        np.real(Wad),
        np.real(Wback),
        np.real(Wc),
        beta_tilde_val,
        beta_line_val,
        beta_star_val,
        D_star_val,
        D_tilde_val,
        Wise,
        np.real(sz_avg),
        np.real(Delta_Svol)
    )


# ---------------- OUTPUT ----------------
shape = (len(n_points), len(taupoints))

Dad_points = np.zeros(shape)
W_points = np.zeros(shape)
Wad_points = np.zeros(shape)
Wback_points = np.zeros(shape)
Wc_points = np.zeros(shape)

beta_tilde = np.zeros(shape)
beta_line = np.zeros(shape)
beta_star = np.zeros(shape)

D_star = np.zeros(shape)
D_tilde = np.zeros(shape)

Wise_points = np.zeros(shape)
sz_sum_avg = np.zeros(shape)
Delta_Svol_points = np.zeros(shape)

output_dir = f"results_beta{beta}_L{L}"

os.makedirs(output_dir, exist_ok=True)


# ---------------- MAIN ----------------
def main():

    for q, n in enumerate(n_points):

        print(f"\nRunning n = {n}")

        S_z, H_static = build_operators(n)

        H0 = Ham(0, 1, S_z, H_static)

        rho0 = rho_eq(H0, beta)

        # Parallel tau calculations
        results = Parallel(n_jobs=N_JOBS)(
            delayed(compute_tau)(
                j,
                tau,
                rho0,
                S_z,
                H_static,
                n
            )
            for j, tau in enumerate(taupoints)
        )

        # Store results
        for res in results:

            (
                j,
                D_ad,
                W,
                Wad,
                Wback,
                Wc,
                beta_tilde_val,
                beta_line_val,
                beta_star_val,
                D_star_val,
                D_tilde_val,
                Wise,
                sz_avg,
                Delta_Svol
            ) = res

            Dad_points[q, j] = D_ad
            W_points[q, j] = W
            Wad_points[q, j] = Wad
            Wback_points[q, j] = Wback
            Wc_points[q, j] = Wc

            beta_tilde[q, j] = beta_tilde_val
            beta_line[q, j] = beta_line_val
            beta_star[q, j] = beta_star_val

            D_star[q, j] = D_star_val
            D_tilde[q, j] = D_tilde_val

            Wise_points[q, j] = Wise
            sz_sum_avg[q, j] = sz_avg
            Delta_Svol_points[q, j] = Delta_Svol

    # ---------------- SAVE DATA ----------------
    np.savetxt(f'{output_dir}/D_ad.txt', Dad_points)
    np.savetxt(f'{output_dir}/W.txt', W_points)
    np.savetxt(f'{output_dir}/W_ad.txt', Wad_points)

    np.savetxt(f'{output_dir}/W_back.txt', Wback_points)
    np.savetxt(f'{output_dir}/Wc.txt', Wc_points)

    np.savetxt(f'{output_dir}/beta_tilde.txt', beta_tilde)
    np.savetxt(f'{output_dir}/beta_line.txt', beta_line)
    np.savetxt(f'{output_dir}/beta_star.txt', beta_star)

    np.savetxt(f'{output_dir}/D_star.txt', D_star)
    np.savetxt(f'{output_dir}/D_tilde.txt', D_tilde)

    np.savetxt(f'{output_dir}/Sz_sum.txt', sz_sum_avg)
    np.savetxt(f'{output_dir}/W_ise.txt', Wise_points)
    np.savetxt(f'{output_dir}/DeltaS_vol.txt', Delta_Svol_points)

    print("\nAll calculations finished.")


if __name__ == "__main__":
    main()
