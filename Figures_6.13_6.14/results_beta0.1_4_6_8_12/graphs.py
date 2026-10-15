import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

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
def rho_eq(evals, evecs, beta):

    rho = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T

    return rho / np.trace(rho)


def rho_eq_diag(beta, evals):

    p = np.exp(-beta * evals)

    return p / np.sum(p)


def rho_eq_tau(beta, evals, evecs):

    rho = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T

    return rho / np.trace(rho)



taupoints = np.logspace(-1, 3, 12)
n_points = [6, 8, 12]
beta = 0.1
J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25
h_0 = hz0
DELTA = Delta_h


Sz_0 = []
Sz_ad = []
for q, n in enumerate(n_points):
    tau = 1.0
    S_z, H_static = build_operators(n)

    H0 = H_static + h(0, tau) * S_z
    Hf = H_static + h(tau, tau) * S_z

    evals0, evecs0 = sl.eigh(H0)
    evalsf, evecsf = sl.eigh(Hf)

    Sz_0.append(np.trace(rho_eq(evals0, evecs0, beta) @ S_z))

    # Adiabatic state
    prbs = np.exp(-beta * evals0)
    prbs /= prbs.sum()
    r_ad = evecsf @ np.diag(prbs) @ evecsf.conj().T

    Sz_ad.append(np.trace(r_ad @ S_z))



DeltaS_points = np.real(np.loadtxt(f'D_tilde.txt', complex))[1:, :]
DeltaS_star = np.real(np.loadtxt(f'D_star.txt', complex))[1:, :]
Dad_points = np.real(np.loadtxt(f'D_ad.txt', complex))[1:, :]
W_points = np.real(np.loadtxt(f'W.txt', complex))[1:, :]
Wad_points = np.real(np.loadtxt(f'W_ad.txt', complex))[1:, :]
Wex_points = W_points - Wad_points
Delta_Svol = np.real(np.loadtxt(f'DeltaS_vol.txt', complex))[1:, :]
sz_sum_avg = np.real(np.loadtxt(f'Sz_sum.txt', complex))[1:, :]
Wise_points = np.real(np.loadtxt(f'W_ise.txt', complex))[1:, :]
W_th = W_points - Wise_points
W_th_ad = Wad_points - Wise_points




fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(DeltaS_points[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(DeltaS_star[0,:]*np.ones(len(taupoints))), 'r--')
ax.loglog(taupoints, np.real(DeltaS_points[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(DeltaS_star[1]*np.ones(len(taupoints))), 'k--')
ax.loglog(taupoints, np.real(DeltaS_points[2,:]), 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(DeltaS_star[2]*np.ones(len(taupoints))), 'b--')
#ax.loglog(taupoints, np.real(DeltaS_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
#ax.loglog(taupoints, np.real(DeltaS_star[3]*np.ones(len(taupoints))), 'g--')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$\Delta \tilde S$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/DeltaS_tilde beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Dad_points[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Dad_points[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Dad_points[2,:]), 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Dad_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$D_{ad}$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/non_ad beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Wex_points[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Wex_points[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Wex_points[2,:]), 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Wex_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$J^{-1}W_{ex}$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Wex beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Delta_Svol[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Delta_Svol[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Delta_Svol[2,:]), 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$\Delta S_{vol}$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/DeltaS_vol beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(W_th[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(W_th_ad[0,:]), 'r--')
ax.loglog(taupoints, np.real(W_th[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(W_th_ad[1,:]), 'k--')
ax.loglog(taupoints, np.real(W_th[2,:]), 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(W_th_ad[2,:]), 'b--')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$W_{ex}^{th}/J$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Wex_th beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(-sz_sum_avg[0,:]), 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(-Sz_ad[0])*np.ones_like(taupoints), 'r--')
ax.loglog(taupoints, np.real(-sz_sum_avg[1,:]), 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(-Sz_ad[1])*np.ones_like(taupoints), 'k--')
ax.loglog(taupoints, np.real(-sz_sum_avg[2,:]), 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(-Sz_ad[2])*np.ones_like(taupoints), 'b--')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower right', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$-\langle \sum_j \sigma_j^z \rangle$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Sz_sum_avg beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()




#### GRAPHS DIVIDED BY L ####


n0, n1, n2 = n_points

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(DeltaS_points[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(DeltaS_star[0,:]*np.ones(len(taupoints))/n0), 'r--')
ax.loglog(taupoints, np.real(DeltaS_points[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(DeltaS_star[1]*np.ones(len(taupoints))/n1), 'k--')
ax.loglog(taupoints, np.real(DeltaS_points[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(DeltaS_star[2]*np.ones(len(taupoints))/n2), 'b--')
#ax.loglog(taupoints, np.real(DeltaS_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
#ax.loglog(taupoints, np.real(DeltaS_star[3]*np.ones(len(taupoints))), 'g--')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$\Delta \tilde S/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/DeltaS_tilde_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Dad_points[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Dad_points[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Dad_points[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Dad_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/non_ad_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Wex_points[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Wex_points[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Wex_points[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Wex_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$J^{-1}W_{ex}/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Wex_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Delta_Svol[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Delta_Svol[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Delta_Svol[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$\Delta S_{vol}/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/DeltaS_vol_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(W_th[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(W_th_ad[0,:])/n0, 'r--')
ax.loglog(taupoints, np.real(W_th[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(W_th_ad[1,:])/n1, 'k--')
ax.loglog(taupoints, np.real(W_th[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(W_th_ad[2,:])/n2, 'b--')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$W_{ex}^{th}/(LJ)$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Wex_th_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(-sz_sum_avg[0,:])/n0, 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(-Sz_ad[0])*np.ones_like(taupoints)/n0, 'r--')
ax.loglog(taupoints, np.real(-sz_sum_avg[1,:])/n1, 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(-Sz_ad[1])*np.ones_like(taupoints)/n1, 'k--')
ax.loglog(taupoints, np.real(-sz_sum_avg[2,:])/n2, 'b.', label=f'$L = {n_points[2]}$')
ax.loglog(taupoints, np.real(-Sz_ad[2])*np.ones_like(taupoints)/n2, 'b--')
#ax.loglog(taupoints, np.real(Delta_Svol[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower right', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$-\langle \sum_j \sigma_j^z \rangle/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\12spins\chain_size _comparison\beta0,1\6_8_12/Sz_sum_avg_div_L beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
#plt.show()