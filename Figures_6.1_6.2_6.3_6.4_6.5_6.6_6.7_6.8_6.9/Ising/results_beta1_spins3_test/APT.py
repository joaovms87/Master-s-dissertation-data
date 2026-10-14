import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import cmath
from numpy.linalg import eigh, eigvalsh
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


h0 = 1.1
h2 = 1
h3 = 1
DELTA = 1  # total variation of h


def h(t):  # protocol for B variation
    if t > 0:
        return h0 + DELTA/tau*t
    else:
        return h0

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


def build_operators(N):

    dim = 2**N

    S_z = np.zeros((dim, dim), dtype=np.complex128)
    S_z0 = np.zeros((dim, dim), dtype=np.complex128)
    S_xx = np.zeros((dim, dim), dtype=np.complex128)

    for j in range(N):
        ops = [Id] * N
        ops[j] = sigma_z
        if j==0:
            S_z0 += tensor(ops)
        else:
            S_z += h_static[j] * tensor(ops)

    for j in range(N - 1):
        ops = [Id] * N
        ops[j], ops[j + 1] = sigma_x, sigma_x
        S_xx += J * tensor(ops)

    H_static = S_xx + S_z

    return S_z0, H_static

def Ham(t):
    return h(t) * S_z0 + H_static


def HAM(lamda):
    return lamda * S_z0 + H_static


def E(m, n, t):
    eigenvalues = eigvalsh(Ham(t))
    return eigenvalues[m] - eigenvalues[n]



S_z0, H_static = build_operators(3)

Del_H = S_z0


####################### FUNCTIONS AND PARAMETERS FOR APT ##########################################################

ind = 0  # (geometric phase will not be calculated because it is zero)
# ind =! 0 if geometric phase =! 0

n_dim = 8  # Dimension of the Hilbert space

def ratio_M_E(m, n, t):
    eigenvalues, eigenvectors = eigh(Ham(t))
    E_mn = eigenvalues[m] - eigenvalues[n]

    for w in range(n_dim):
        if sum(eigenvectors[:, w]) < 0:
            eigenvectors[:, w] = -eigenvectors[:, w]

    matrix_element = np.conjugate(eigenvectors[:, m]) @ Del_H @ eigenvectors[:, n]
    if m != n:
        return -DELTA/tau * matrix_element / E_mn**2
    else:
        return 0

def M(m, n, t):
    if m==n:
        return M_diag(n, h(t), ind)
    else:
        eigenvalues, eigenvectors = eigh(Ham(t))

        for w in range(n_dim):
            if sum(eigenvectors[:, w]) < 0:
                eigenvectors[:, w] = -eigenvectors[:, w]

        E_mn = eigenvalues[m] - eigenvalues[n]
        matrix_element = np.conjugate(eigenvectors[:, m]) @ Del_H @ eigenvectors[:, n]
        return -DELTA/tau * matrix_element / E_mn


def omega(m, n, t):

    def integrand(x):
        eigenvalues = eigvalsh(Ham(x))
        return np.real(eigenvalues[m] - eigenvalues[n])

    return quad(integrand, 0, t)[0]


def M_diag(n, h1, ind):
    if ind == 0:
        return 0
    else:
        delt = 1e-6
        eigs, eigenvectors = eigh(HAM(h1))
        eigs_m, eigenvectors_m = eigh(HAM(h1-delt))
        eigs_p, eigenvectors_p = eigh(HAM(h1+delt))

        """
        for i in range(8):
            for j in range(8):
                if abs(eigenvectors_m[i, j]) >= 1e-12:
                    if -1.001 < eigenvectors_p[i, j]/eigenvectors_m[i, j] < -0.999:
                        eigenvectors_m[i, j] = - eigenvectors_m[i, j]
        """

        for w in range(n_dim):
            if sum(eigenvectors[:, w]) < 0:
                eigenvectors[:, w] = -eigenvectors[:, w]
            if sum(eigenvectors_m[:, w]) < 0:
                eigenvectors_m[:, w] = -eigenvectors_m[:, w]
            if sum(eigenvectors_p[:, w]) < 0:
                eigenvectors_p[:, w] = -eigenvectors_p[:, w]

        d_eigen = (eigenvectors_p[:, n] - eigenvectors_m[:, n])/delt

        return np.vdot(eigenvectors[:, n], d_eigen)


def gamma(m, n, h1):

    def integrand_g(x):
        return np.real(1j*(M_diag(m, x, ind) - M_diag(n, x, ind)))

    I = quad(integrand_g, h0, h1, epsabs=1e-16, epsrel=1e-16)
    print(I)

    return I[0]


def phi(m, n, t, ind):
    if ind == 0:
        return -omega(m, n, t)
    else:
        return gamma(m, n, h(t)) - omega(m, n, t)


def C1(m, n, t):
    return 1j*(ratio_M_E(m, n, t) - cmath.exp(1j*phi(m, n, t, ind))*ratio_M_E(m, n, 0))


def p(k, b):
    eigenvalues = eigvalsh(Ham(0))
    Z = sum(np.exp(-b*eigenvalues))
    return np.exp(-b*eigenvalues[k])/Z


for beta in [1]:
    Wpoints = np.loadtxt(f'W.txt', complex)
    D2points = np.real(np.loadtxt(f'D_ad.txt', complex))
    WADpoints = np.loadtxt(f'W_ad.txt', complex)
    WexNUM = np.real(Wpoints-WADpoints)
    zzzz = len(Wpoints)
    # list of the process durations that will be investigated
    taupoints = np.logspace(-1,3, zzzz)
    WexAPT = np.empty(zzzz, float)
    D2APT = np.empty(zzzz, float)


    for i, tau in enumerate(taupoints):
        Wex = 0
        D = 0
        for m in range(0, 8):
            for n in range(0, 8):
                D += p(n, beta)*E(m, n, 0)*abs(C1(m, n, tau))**2
                Wex += p(n, beta)*E(m, n, tau)*abs(C1(m, n, tau))**2
        WexAPT[i] = Wex
        D2APT[i] = beta*D
    ratio = D2APT/WexAPT
    #print(ratio)
    avg_ratio = np.average(ratio)


    print(abs(WexAPT - WexNUM)/WexAPT)

    # Wex comparison: 10^(-2) - 10^4

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, WexNUM, 'r.', label=r'$W_{ex}/J$ Num. solution')
    ax.loglog(taupoints, WexAPT, 'k.', label=r'$W_{ex}/J$ $2^{nd}$ order APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    #ax.set_ylabel('$W_{ex}/J$')
    #ax.set_title(r'$\tau$-dependence of the excess work')
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    #ax.set_box_aspect(1)
    ax.set_box_aspect(0.85)
    plt.savefig(f'D:\MESTRADO\integrable ising\spins3_1varia/log(Wex)_log(tau),h0={h0},DELTA={DELTA}.png', bbox_inches='tight')



    # D_ad comparison: 10^(-2) - 10^4
"""
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, D2points, 'r.', label=r'Num. solution')
    ax.loglog(taupoints, D2APT, 'k.', label=r'$2^{nd}$ order APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('Relative entropy')
    ax.set_title(r'$\tau$-dependence of $D_{\tau ,ad}(\tau)$')
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\MESTRADO\Comparação_Num_vs_APT\spins3_1varia\BETA={beta}/log(D_ad)_log(tau),h0={h0},DELTA={DELTA}.png', bbox_inches='tight')


    # To second order, APT predicts proportionality between D2 and Wex.
    # Here, we compare the numerical results of D2 with the multiplication of WexNUM by the predicted constant.
    APT_D2 = avg_ratio*WexNUM
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, D2points, 'r.', label=r'Num. solution $D_{\tau,ad}^{(num)}(\tau)$')
    ax.loglog(taupoints, APT_D2, 'k.', label=r'$(predicted \ ratio)\cdot W_{\tau,ex}^{(num)}(\tau)$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('Relative entropy')
    ax.set_title(r'Are $D_{\tau ,ad}(\tau)$ and $W_{\tau ,ex}(\tau)$ proportional?')
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(1)
    plt.savefig(f'D:\MESTRADO\Comparação_Num_vs_APT\spins3_1varia\BETA={beta}/proporcionalidade,h0={h0},DELTA={DELTA}.png', bbox_inches='tight')


    ############################################# EQM gráficos ########################################################

    plt.rcParams['errorbar.capsize'] = 3
    # plt.rcParams['mathtext.fontset'] = 'cm'
    # plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'medium'

    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, WexNUM, 'r.', label=r'$W_{ex}(\tau)$ num.')
    ax.loglog(taupoints, WexAPT, 'k.', label=r'$W_{ex}(\tau)$ APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel('$W_{ex} J^{-1}$')
    ax.set_title(r'3 spins: $W_{ex}(\tau)$ numérico vs APT', pad=10)
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(0.7)
    plt.savefig(f'D:\MESTRADO\EQM\spins3 Wex comparação APT h0={h0}.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, D2points, 'r.', label=r'$D_{ad}(\tau)$ num.')
    ax.loglog(taupoints, D2APT, 'k.', label=r'$D_{ad}(\tau)$ APT')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel(r'$D_{ad}$')
    ax.set_title(r'LZ: $D_{ad}(\tau)$ numérico vs APT', pad=10)
    # ax.set_ylim(min(WexNUM), 1.5*max(WexNUM))
    ax.set_box_aspect(0.7)
    plt.savefig(f'D:\MESTRADO\EQM/spins3 Dad comparação APT h0={h0}.png', bbox_inches='tight')
"""