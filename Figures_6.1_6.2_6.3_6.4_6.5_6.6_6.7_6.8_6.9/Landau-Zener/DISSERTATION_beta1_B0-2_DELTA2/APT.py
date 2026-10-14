import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cmath

plt.rcParams['errorbar.capsize'] = 3
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.rm'] = 'serif'
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

beta = 1
DELTA = 2  # total variation of B


def B(t):  # protocol for B variation
    if t > 0:
        return B0 + DELTA / tau * t
    else:
        return B0


def theta(B):  # a useful parameter dependent on B
    return 0.5 * np.arctan2(1, B)


def E(B):  # absolute value of the eigen-energy of the system
    return np.sqrt(B ** 2 + 1)


def ratio_M_E(t):
    return -DELTA / tau / 4 / E(B(t)) ** 3


def integrand(t):
    return np.sqrt(1 + B(t) ** 2)


def phi(m, n, t):
    return (m - n) * 2 * quad(integrand, 0, t, epsabs=1e-12, epsrel=1e-12)[0]


def C1(m, n, t):
    return 1j * (ratio_M_E(t) - cmath.exp(1j * phi(m, n, t)) * ratio_M_E(0))


def avg_abs_squared_C1(m, n, t):
    return abs(ratio_M_E(t)) ** 2 + abs(ratio_M_E(0)) ** 2 - 2 * avg_cos * abs(ratio_M_E(t) * ratio_M_E(0))


B0 = -2
Wpoints = np.loadtxt(f'W.txt', complex)
D2points = np.real(np.loadtxt(f'D_ad.txt', complex))
WADpoints = np.loadtxt(f'W_ad.txt', complex)
WexNUM = np.real(Wpoints - WADpoints)
n = len(Wpoints)
# list of the process durations that will be investigated
taupoints = np.logspace(-1, 3, n)
WexAPT = np.empty(n, float)
WexAPT_avg = np.empty(n, float)
D2APT = np.empty(n, float)
D2APT_avg = np.empty(n, float)

phi_list = np.empty(int(1e3), float)
sample_points = np.linspace(1e-1, 1e3, int(1e3))
for k, s in enumerate(sample_points):
        tau = s
        phi_list[k] = phi(2, 1, s)
cosine = np.cos(phi_list)
avg_cos = np.average(cosine)

for i, tau in enumerate(taupoints):
        C = C1(2, 1, tau)
        Wex = 2 * E(B(tau)) * np.tanh(beta * E(B0)) * (abs(C)) ** 2
        D = 2 * beta * E(B0) * np.tanh(beta * E(B0)) * (abs(C)) ** 2
        WexAPT[i] = Wex
        D2APT[i] = D

        C2_avg = avg_abs_squared_C1(2, 1, tau)
        Wex_avg = 2 * E(B(tau)) * np.tanh(beta * E(B0)) * C2_avg
        WexAPT_avg[i] = Wex_avg
        D_avg = 2 * beta * E(B0) * np.tanh(beta * E(B0)) * C2_avg
        D2APT_avg[i] = D_avg

# Wex comparison: 10^(-2) - 10^4

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, WexNUM, 'r.', label=r'Num. solution')
ax.loglog(taupoints, WexAPT, 'k.', label=r'$2^{nd}$ order APT')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W_{ex}/J$')
#ax.set_title(r'$\tau$-dependence of the excess work')
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
#ax.set_box_aspect(1)
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION/log(Wex)_log(tau),B0=-2,beta=1.png',
                bbox_inches='tight')

# Wex comparison: averaging the phase
"""
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, WexNUM, 'r.', label=r'Num. solution')
ax.loglog(taupoints, WexAPT_avg, 'k-', label=r'$2^{nd}$ order APT (phase-averaged)')
ax.legend(fancybox=False, edgecolor='black', loc='upper right')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('$W \cdot J^{-1}$')
ax.set_title(r'$\tau$-dependence of the excess work')
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
#ax.set_box_aspect(1)
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION/log(Wex)_log(tau) phase-averaged,B0=-2,beta=1.png',
        bbox_inches='tight')
"""

# D_ad comparison: 10^(-2) - 10^4

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, D2points, 'r.', label=r'Num. solution')
ax.loglog(taupoints, D2APT, 'k.', label=r'$2^{nd}$ order APT')
ax.legend(fancybox=False, edgecolor='black', loc='lower left')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('Relative entropy')
#ax.set_title(r'$\tau$-dependence of $D_{\tau ,ad}(\tau)$')
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
#ax.set_box_aspect(1)
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION/log(D_ad)_log(tau),B0=-2,beta=1.png',
                bbox_inches='tight')

# D_ad comparison: averaging the phase
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, D2points, 'r.', label=r'Num. solution')
ax.loglog(taupoints, D2APT_avg, 'k-', label=r'$2^{nd}$ order APT (phase-averaged)')
ax.legend(fancybox=False, edgecolor='black', loc='upper right')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('Relative entropy')
#ax.set_title(r'$\tau$-dependence of $D_{\tau ,ad}(\tau)$')
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
#ax.set_box_aspect(1)
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION/log(D_ad)_log(tau) phase-averaged,B0=-2,beta=1.png',
        bbox_inches='tight')

"""

    # Comparison: 10^3 - 10^4

    br = 10**3.98
    for i, tau in enumerate(taupoints):
        if tau > br:
            break

    x = np.logspace(3.98, 4, 1000)
    WexAPT = np.empty(1000)
    for j, tau in enumerate(x):
        C = C1(2, 1, tau)
        Wex = 2*E(B(tau))*np.tanh(beta*E(B0))*(abs(C))**2
        WexAPT[j] = Wex

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints[i:], WexNUM[i:], 'r.', label=r'Numerical solution')
    ax.loglog(x, WexAPT, 'k-', label=r'First order APT')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'$\tau$-dependence of the excess work')
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_xlim(10**3.98, 1e4)
    ax.set_box_aspect(1)
    plt.savefig(f'D:\MESTRADO\Comparação_Num_vs_APT\B0={B0}/log(Wex)_adiabatic,B0={B0},beta={beta}.png', bbox_inches='tight')



# To second order, APT predicts proportionality between D2 and Wex.
# Here, we compare the numerical results of D2 with the multiplication of WexNUM by the predicted constant.
APT_D2 = beta * E(B0) / E(B0 + DELTA) * WexNUM
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, D2points, 'r.', label=r'Num. solution $D_{\tau,ad}^{(num)}(\tau)$')
ax.loglog(taupoints, APT_D2, 'k.', label=r'$\beta\frac{E(B_0)}{E(B_\tau)}W_{\tau,ex}^{(num)}(\tau)$')
ax.legend(fancybox=False, edgecolor='black')
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel('Relative entropy')
ax.set_title(r'Are $D_{\tau ,ad}(\tau)$ and $W_{\tau ,ex}(\tau)$ proportional?')
# ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
ax.set_box_aspect(1)
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION/proporcionalidade,B0=-2,beta=1.png',
                bbox_inches='tight')
"""

print(abs(WexNUM - WexAPT)/WexNUM)