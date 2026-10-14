import numpy as np
import scipy.linalg as sl
from scipy.optimize import fsolve
from scipy.stats import entropy
from joblib import Parallel, delayed
import os
from scipy.integrate import solve_ivp




# ---------------- PARAMETERS ----------------
beta = 0.1
J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25

h_0 = hz0
DELTA = Delta_h

n_points = [8]
taupoints = np.logspace(-1, 4, 20)

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
        u = 0.005

    def f(r, t):

        H = H_static + h(t, tau) * S_z

        return -1j * (H @ r - r @ H)

    while t < tau:

        k1 = u * f(r, t)
        k2 = u * f(r + 0.5 * k1, t + 0.5 * u)
        k3 = u * f(r + 0.5 * k2, t + 0.5 * u)
        k4 = u * f(r + k3, t + u)

        r = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t += u

    return r

"""
def evolve(rho0, S_z, H_static, tau):

    dim = rho0.shape[0]

    # flatten density matrix into vector
    y0 = rho0.reshape(dim * dim)

    def rhs(t, y):

        # reconstruct density matrix
        rho = y.reshape((dim, dim))

        H = H_static + h(t, tau) * S_z

        drho = -1j * (H @ rho - rho @ H)

        # flatten again for solve_ivp
        return drho.reshape(dim * dim)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, tau),
        y0=y0,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11
    )

    # final state
    rho_final = sol.y[:, -1].reshape((dim, dim))

    return rho_final
"""


# ---------------- SINGLE TAU COMPUTATION ----------------
def compute_tau(j, tau, rho0, S_z, H_static, n):

    print(f"Starting tau = {tau:.3e}")

    rho = evolve(rho0, S_z, H_static, tau)

    Hf = H_static + h(tau, tau) * S_z

    evals, evecs = sl.eigh(Hf)

    H0 = H_static + h(0, tau) * S_z

    eigvals0, eigvecs0 = sl.eigh(H0)

    # Adiabatic state
    prbs = np.exp(-beta * eigvals0)
    prbs /= prbs.sum()

    r_ad = evecs @ np.diag(prbs) @ evecs.conj().T

    ln_rho_ad = evecs @ np.diag(np.log(prbs)) @ evecs.conj().T

    E0 = np.trace(H0 @ rho0)

    Wad = np.dot(prbs, evals) - E0

    W = np.trace(Hf @ rho) - E0

    # Temperatures
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

    # Divergences
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
        np.sum(rho_eq_diag(beta_line_val, evals) * evals)
        - E0
    )


    n_measurements = 10

    Mx_list = []
    Mx_list0 = []

    edges = np.linspace(evals[0], evals[-1], n_measurements + 1)
    for i in range(n_measurements):

        if i < n_measurements - 1:
            mask = (evals >= edges[i]) & (evals < edges[i + 1])
        else:
            mask = (evals >= edges[i]) & (evals <= edges[i + 1])

        M = np.zeros_like(Hf)

        for k in np.where(mask)[0]:
            M += np.outer(evecs[:, k], evecs[:, k].conj())

        Mx_list.append(M)

    edges = np.linspace(eigvals0[0], eigvals0[-1], n_measurements + 1)
    for i in range(n_measurements):

        if i < n_measurements - 1:
            mask = (eigvals0 >= edges[i]) & (eigvals0 < edges[i + 1])
        else:
            mask = (eigvals0 >= edges[i]) & (eigvals0 <= edges[i + 1])

        M = np.zeros_like(H0)

        for k in np.where(mask)[0]:
            M += np.outer(eigvecs0[:, k], eigvecs0[:, k].conj())

        Mx_list0.append(M)

    """
    Id_check = np.sum(Mx_list, axis=0)
    print(np.linalg.norm(Id_check - np.eye(len(evals))))

    Id_check = np.sum(Mx_list0, axis=0)
    print(np.linalg.norm(Id_check - np.eye(len(evals))))
    """

    coarse_p_rho = []
    coarse_p_prior = []
    coarse_p0 = []
    coarse_Id = []
    coarse_Id0 = []
    for i in range(n_measurements):
        Mx = Mx_list[i]
        Mx_0 = Mx_list0[i]
        coarse_p_rho.append(np.trace(Mx @ rho))
        coarse_p_prior.append(np.trace(Mx @ rho_eq_tau(beta_tilde_val, evals, evecs)))
        coarse_p0.append(np.trace(Mx_0 @ rho0))

        coarse_Id.append(np.trace(Mx))
        coarse_Id0.append(np.trace(Mx_0))

    coarse_p_rho = np.real(np.array(coarse_p_rho))
    coarse_p_prior = np.real(np.array(coarse_p_prior))
    coarse_p0 = np.real(np.array(coarse_p0))

    p_tilde = np.real(np.exp(-beta_tilde_val * evals))
    p_tilde = p_tilde / sum(p_tilde)
    p_rho = np.real(np.diag(evecs.conj().T @ rho @ evecs))

    p_0 = np.real(np.exp(-beta * eigvals0))

    coarse_Id = np.array(coarse_Id, dtype=float)
    coarse_Id0 = np.array(coarse_Id0, dtype=float)

    """
    print(type(coarse_Id))
    print(type(coarse_Id0))

    print(np.array(coarse_Id).dtype)
    print(np.array(coarse_Id0).dtype)

    print(np.array(coarse_p_rho).dtype)
    print(np.array(coarse_p0).dtype)
    """



    obs_entropy = entropy(p_tilde) - entropy(p_rho, p_tilde) - entropy(p_0)

    diag_entropy = entropy(p_rho) - entropy(p_0)

    p_ones = np.ones(len(p_rho)) / len(p_rho)
    obs_entropy_traditional = - entropy(p_rho, p_ones) + entropy(p_0, p_ones)

    Sobs_coarse = entropy(p_tilde) - entropy(coarse_p_rho, coarse_p_prior) - entropy(p_0)
    Strad_coarse = - entropy(coarse_p_rho, coarse_Id) + entropy(coarse_p0, coarse_Id0)


    # Observable
    sz_avg = np.trace(rho @ S_z)

    # Entropy production
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
        beta_tilde_val,
        beta_line_val,
        beta_star_val,
        D_star_val,
        D_tilde_val,
        Wise,
        np.real(sz_avg),
        np.real(Delta_Svol),
        np.real(obs_entropy),
        np.real(obs_entropy_traditional),
        np.real(Sobs_coarse),
        np.real(Strad_coarse),
        np.real(diag_entropy)
    )


# ---------------- OUTPUT ----------------
shape = (len(n_points), len(taupoints))

Dad_points = np.zeros(shape)
W_points = np.zeros(shape)
Wad_points = np.zeros(shape)
beta_tilde = np.zeros(shape)
beta_line = np.zeros(shape)
beta_star = np.zeros(shape)
D_star = np.zeros(shape)
D_tilde = np.zeros(shape)
Wise_points = np.zeros(shape)
sz_sum_avg = np.zeros(shape)
Delta_Svol_points = np.zeros(shape)
Sobs_fine_points = np.zeros(shape)
Sobs_trad_fine_points = np.zeros(shape)
Sobs_coarse_points = np.zeros(shape)
Sobs_trad_coarse_points = np.zeros(shape)
Sd_points = np.zeros(shape)



output_dir = f"results_beta{beta} with coarse entropies"

os.makedirs(output_dir, exist_ok=True)


# ---------------- MAIN ----------------
def main():

    for q, n in enumerate(n_points):

        print(f"\nRunning n = {n}")

        S_z, H_static = build_operators(n)

        H0 = H_static + h(0, 1) * S_z

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
                beta_tilde_val,
                beta_line_val,
                beta_star_val,
                D_star_val,
                D_tilde_val,
                Wise,
                sz_avg,
                Delta_Svol,
                obs_entropy,
                obs_entropy_traditional,
                Sobs_coarse,
                Strad_coarse,
                diag_entropy
            ) = res

            Dad_points[q, j] = D_ad
            W_points[q, j] = W
            Wad_points[q, j] = Wad
            beta_tilde[q, j] = beta_tilde_val
            beta_line[q, j] = beta_line_val
            beta_star[q, j] = beta_star_val
            D_star[q, j] = D_star_val
            D_tilde[q, j] = D_tilde_val
            Wise_points[q, j] = Wise
            sz_sum_avg[q, j] = sz_avg
            Delta_Svol_points[q, j] = Delta_Svol
            Sobs_fine_points[q, j] = obs_entropy
            Sobs_trad_fine_points[q, j] = obs_entropy_traditional
            Sobs_coarse_points[q, j] = Sobs_coarse
            Sobs_trad_coarse_points[q, j] = Strad_coarse
            Sd_points[q, j] = diag_entropy

    # Save data
    np.savetxt(f'{output_dir}/D_ad.txt', Dad_points)
    np.savetxt(f'{output_dir}/W.txt', W_points)
    np.savetxt(f'{output_dir}/W_ad.txt', Wad_points)
    np.savetxt(f'{output_dir}/beta_tilde.txt', beta_tilde)
    np.savetxt(f'{output_dir}/beta_line.txt', beta_line)
    np.savetxt(f'{output_dir}/beta_star.txt', beta_star)
    np.savetxt(f'{output_dir}/D_star.txt', D_star)
    np.savetxt(f'{output_dir}/D_tilde.txt', D_tilde)
    np.savetxt(f'{output_dir}/Sz_sum.txt', sz_sum_avg)
    np.savetxt(f'{output_dir}/W_ise.txt', Wise_points)
    np.savetxt(f'{output_dir}/DeltaS_vol.txt', Delta_Svol_points)
    np.savetxt(f'{output_dir}/Sobs_fine.txt', Sobs_fine_points)
    np.savetxt(f'{output_dir}/Sobs_trad.txt', Sobs_trad_fine_points)
    np.savetxt(f'{output_dir}/Sobs_coarse.txt', Sobs_coarse_points)
    np.savetxt(f'{output_dir}/Strad_coarse.txt', Sobs_trad_coarse_points)
    np.savetxt(f'{output_dir}/Sd_points.txt', Sd_points)


    print("\nAll calculations finished.")


if __name__ == "__main__":
    main()
