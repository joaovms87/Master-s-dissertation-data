import numpy as np
import scipy.linalg as sl
from fontTools.ttLib.tables.TupleVariation import DELTAS_ARE_LONGS
from scipy.optimize import fsolve
from scipy.stats import entropy
from joblib import Parallel, delayed
import os
from scipy.integrate import solve_ivp




# ---------------- PARAMETERS ----------------
DELTA = 1  # total variation of gamma1
beta = 1
h2 = 1
h3 = 1
h_static = [0, h2, h3]
J = 1

hz0 = 1.1

h_0 = hz0

n_points = [3]
taupoints = np.logspace(-1, 3, 40)

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

def D(rho1, rho2):
    return np.trace(rho1 @ sl.logm(rho1)) - np.trace(rho1 @ sl.logm(rho2))


# ---------------- PRECOMPUTE OPERATORS ----------------
def build_operators(n):

    dim = 2**n

    S_z = np.zeros((dim, dim), dtype=np.complex128)
    S_z0 = np.zeros((dim, dim), dtype=np.complex128)
    S_xx = np.zeros((dim, dim), dtype=np.complex128)

    for j in range(n):
        ops = [Id] * n
        ops[j] = sigma_z
        if j==0:
            S_z0 += tensor(ops)
        else:
            S_z += h_static[j] * tensor(ops)

    for j in range(n - 1):
        ops = [Id] * n
        ops[j], ops[j + 1] = sigma_x, sigma_x
        S_xx += J * tensor(ops)

    H_static = S_xx + S_z

    return S_z0, H_static


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
        u = 0.001

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
        rtol=1e-15,
        atol=1e-15
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

    DeltaE_line_val = np.trace((r_ad - rho_eq_tau(beta_line_val, evals, evecs)) @ Hf)


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

    p_final = np.einsum(
        'in,ij,jn->n',
        evecs.conj(),
        rho,
        evecs,
        optimize=True
    ).real

    Sd_final = entropy(p_final)

    DeltaS_d = Sd_final - entropy(prbs)

    D_th = D(rho, rho_eq_tau(beta_line_val, evals, evecs))

    D_ad_tilde = D(r_ad, rho_eq_tau(beta_tilde_val, evals, evecs))

    D_rho_star = D(rho, rho_eq_tau(beta_star_val, evals, evecs))



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
        DeltaE_line_val,
        Wise,
        np.real(sz_avg),
        np.real(Delta_Svol),
        np.real(DeltaS_d),
        np.real(D_th),
        np.real(D_ad_tilde),
        np.real(D_rho_star)
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
DeltaE_line = np.zeros(shape)
Wise_points = np.zeros(shape)
sz_sum_avg = np.zeros(shape)
Delta_Svol_points = np.zeros(shape)
Delta_Sd_points = np.zeros(shape)

D_th_points = np.zeros(shape)
D_ad_tilde_points = np.zeros(shape)
D_rho_star_points = np.zeros(shape)

output_dir = f"results_beta{beta}_spins{n_points[0]}_test"

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
                DeltaE_line_val,
                Wise,
                sz_avg,
                Delta_Svol,
                DeltaS_d,
                D_th,
                D_ad_tilde,
                D_rho_star
            ) = res

            Dad_points[q, j] = D_ad
            W_points[q, j] = W
            Wad_points[q, j] = Wad
            beta_tilde[q, j] = beta_tilde_val
            beta_line[q, j] = beta_line_val
            beta_star[q, j] = beta_star_val
            D_star[q, j] = D_star_val
            D_tilde[q, j] = D_tilde_val
            DeltaE_line[q, j] = DeltaE_line_val
            Wise_points[q, j] = Wise
            sz_sum_avg[q, j] = sz_avg
            Delta_Svol_points[q, j] = Delta_Svol
            Delta_Sd_points[q, j] = DeltaS_d
            D_th_points[q, j] = D_th
            D_ad_tilde_points[q, j] = D_ad_tilde
            D_rho_star_points[q, j] = D_rho_star

    # Save data
    np.savetxt(f'{output_dir}/D_ad.txt', Dad_points)
    np.savetxt(f'{output_dir}/W.txt', W_points)
    np.savetxt(f'{output_dir}/W_ad.txt', Wad_points)
    np.savetxt(f'{output_dir}/beta_tilde.txt', beta_tilde)
    np.savetxt(f'{output_dir}/beta_line.txt', beta_line)
    np.savetxt(f'{output_dir}/beta_star.txt', beta_star)
    np.savetxt(f'{output_dir}/D_star.txt', D_star)
    np.savetxt(f'{output_dir}/D_tilde.txt', D_tilde)
    np.savetxt(f'{output_dir}/DeltaE_line.txt', DeltaE_line)
    np.savetxt(f'{output_dir}/Sz_sum.txt', sz_sum_avg)
    np.savetxt(f'{output_dir}/W_ise.txt', Wise_points)
    np.savetxt(f'{output_dir}/DeltaS_vol.txt', Delta_Svol_points)
    np.savetxt(f'{output_dir}/DeltaS_d.txt', Delta_Sd_points)
    np.savetxt(f'{output_dir}/D_th.txt', D_th_points)
    np.savetxt(f'{output_dir}/D_ad_tilde.txt', D_ad_tilde_points)
    np.savetxt(f'{output_dir}/D_rho_star.txt', D_rho_star_points)

    print("\nAll calculations finished.")


if __name__ == "__main__":
    main()
