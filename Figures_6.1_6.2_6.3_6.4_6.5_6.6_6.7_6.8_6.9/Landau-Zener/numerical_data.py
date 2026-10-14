import numpy as np
import scipy.linalg as sl
from fontTools.ttLib.tables.TupleVariation import DELTAS_ARE_LONGS
from scipy.optimize import fsolve
from scipy.stats import entropy
from joblib import Parallel, delayed
import os
from scipy.integrate import solve_ivp




# ---------------- PARAMETERS ----------------
DELTA = 2  # total variation of B
beta = 1
J = 1

hz0 = -2

h_0 = hz0

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


# ---------------- PRECOMPUTE OPERATORS ----------------
def build_operators():

    H_static = J*sigma_x
    S_z0 = sigma_z

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
        rtol=1e-13,
        atol=1e-13
    )

    # final state
    rho_final = sol.y[:, -1].reshape((dim, dim))

    return rho_final
"""


# ---------------- OUTPUT ----------------

Dad_points = np.zeros(len(taupoints))
W_points = np.zeros(len(taupoints))
Wad_points = np.zeros(len(taupoints))
beta_tilde = np.zeros(len(taupoints))
beta_line = np.zeros(len(taupoints))
beta_star = np.zeros(len(taupoints))
D_star = np.zeros(len(taupoints))
D_tilde = np.zeros(len(taupoints))
Wise_points = np.zeros(len(taupoints))
sz_sum_avg = np.zeros(len(taupoints))
Delta_Svol_points = np.zeros(len(taupoints))
Delta_Sd_points = np.zeros(len(taupoints))

output_dir = f"DISSERTATION_beta{beta}_B0{h_0}_DELTA{DELTA}"

os.makedirs(output_dir, exist_ok=True)


# ---------------- MAIN ----------------
def main():

        S_z, H_static = build_operators()

        H0 = H_static + h(0, 1) * S_z

        rho0 = rho_eq(H0, beta)

        for j, tau in enumerate(taupoints):
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

                # Observable
                sz_avg = np.trace(rho @ S_z)

                # Entropy production
                w = 2

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

                print(f"Finished tau = {tau:.3e}")

                Dad_points[j] = D_ad
                W_points[j] = W
                Wad_points[j] = Wad
                beta_tilde[j] = beta_tilde_val
                beta_line[j] = beta_line_val
                beta_star[j] = beta_star_val
                D_star[j] = D_star_val
                D_tilde[j] = D_tilde_val
                Wise_points[j] = Wise
                sz_sum_avg[j] = sz_avg
                Delta_Svol_points[j] = Delta_Svol
                Delta_Sd_points[j] = DeltaS_d

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
        np.savetxt(f'{output_dir}/DeltaS_d.txt', Delta_Sd_points)



if __name__ == "__main__":
    main()
