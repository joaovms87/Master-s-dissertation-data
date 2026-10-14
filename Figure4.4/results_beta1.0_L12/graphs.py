import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
import scipy.linalg as sl
plt.rcParams['errorbar.capsize'] = 3
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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



taupoints = np.logspace(-1, 2, 20)
print(taupoints[12])
Wc = np.real(np.loadtxt(f'Wc.txt', complex))
W = np.real(np.loadtxt(f'W.txt', complex))
W_back = Wc - W
D_ad = np.real(np.loadtxt(f'D_ad.txt', complex))
W_ad = np.real(np.loadtxt(f'W_ad.txt', complex))
W_ex = W - W_ad
W_ise = np.real(np.loadtxt(f'W_ise.txt', complex))
Wex_th = W - W_ise

tau = 1.0
Sz, H_static = build_operators(L)
DeltaF = -1/beta*np.log(np.trace(sl.expm(-beta*Ham(tau, tau, Sz, H_static)))/np.trace(sl.expm(-beta*Ham(0, tau, Sz, H_static))))
W_diss = W - DeltaF


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, D_ad/beta, 'r^', label=r'$\mathcal{A}$')
ax.loglog(taupoints, Wc, 'kv', label=r'$\beta W_c$')
ax.loglog(taupoints, W_back, 'b.', label = r'$\beta W_{ad}^{back}$')
ax.loglog(taupoints, -W_ad, 'k--', label = r'$-\beta W_{ad}$')
#ax.loglog(taupoints, W_ex, 'k--', label = r'$W_{ex}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=16)
ax.set_xlabel(r'$\tau J_m/\hbar$')
#ax.set_ylabel('$D$ or $WJ_m^{-1}$')
#ax.set_title(r'Comparison between $D_{ad}(\tau)$ and $D_{rev}(\tau)$')
#ax.set_xlim(1e-1, 1e3)
ax.set_box_aspect(1)
plt.savefig(f'D:\MESTRADO\PAPER\spins12/D_ad = Wc,h_0={h_0},beta={beta}, DELTA={DELTA}.png', bbox_inches='tight', dpi=600)

print(W_ad + W_back)
print()
print(W-DeltaF)

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, W_diss, 'bd', label = r'$W_{diss}$')
ax.loglog(taupoints, Wex_th, 'kP', label=r'$W_{ex}^{th}$')
ax.loglog(taupoints, W_ex, 'r.', label=r'$W_{ex}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=16)
ax.set_xlabel(r'$\tau J_m/\hbar$')
ax.set_ylabel('$W/J_m$')
#ax.set_title(r'Comparison between $D_{ad}(\tau)$ and $D_{rev}(\tau)$')
#ax.set_xlim(1e-1, 1e3)
ax.set_box_aspect(1)
plt.savefig(f'D:\MESTRADO\PAPER\spins12/excess work comparison,h_0={h_0},beta={beta}, DELTA={DELTA}.png', bbox_inches='tight', dpi=600)