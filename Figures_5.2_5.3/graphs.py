import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
plt.rcParams['errorbar.capsize'] = 3
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['grid.color'] = 'k'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'small'

plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


beta = 1
h2 = 1
h3 = 1

taupoints = np.logspace(-1, 2, 50)

for h_0 in [1.1, 2.1]:

    if h_0==1.1:
        DELTA = 1
    else:
        DELTA = -1

    beta_star = np.loadtxt(f'beta_star beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', float)
    D_star = np.loadtxt(f'D_star beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
    Delta_S = np.loadtxt(f'Delta_S beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
    Wex = np.loadtxt(f'Wex beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
    D2 = np.loadtxt(f'D2 beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)

    beta_tilde = np.loadtxt(f'beta_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
    D_tilde = np.loadtxt(f'D_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)
    Delta_S_tilde = np.loadtxt(f'Delta_S_tilde beta={beta} h_0={h_0} h2={h2} h3={h3} DELTA={DELTA}', complex)

    """
    print(beta_star)
    print(Delta_S)
    print()
    """

    # plot D_star and Delta_S in log scale together with Wex
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, np.real(beta_star*Wex), 'rv', label=r'$\beta^*W_{ex}(\tau)$')
    ax.loglog(taupoints, np.real(D_star), 'b.', label=r'$D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$')
    ax.loglog(taupoints, np.real(Delta_S), 'k--', label=r'$\Delta S^*$')
    ax.loglog(taupoints, np.real(D_star - Delta_S), 'k^', label=r'$D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]-\Delta S^*$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel(r'$\beta^*W_{ex} \ $ or $\ \ D$ or $\ \Delta S^*$')
    #ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
    #ax.set_xlim(1e-1, 1e3)
    #ax.set_ylim(top=1)
    ax.set_box_aspect(0.85)
    plt.savefig(f'D:\MESTRADO\spins3_1varia\Beta_star/everything,beta={beta},h_0={h_0},DELTA={DELTA}.png', bbox_inches='tight', dpi=600)

    print(beta_star)
    print(np.real(D2 - beta_star*Wex)/np.real(D2))


    # plot D_tilde and Delta_S_tilde in log scale together with Wex
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.loglog(taupoints, np.real(beta_tilde*Wex), 'rv', label=r'$\tilde \beta W_{ex}(\tau)$')
    ax.loglog(taupoints, np.real(D_tilde), 'b.', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\tilde \beta}(\lambda_\tau)]$')
    ax.loglog(taupoints, np.real(Delta_S_tilde), 'k--', label=r'$\Delta \tilde S$')
    ax.loglog(taupoints, np.real(Delta_S_tilde - D_tilde), 'k^', label=r'$\Delta \tilde S - D[\rho_{ad}(\lambda_\tau)||\Pi_{\tilde \beta}(\lambda_\tau)]$')
    ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=13)
    ax.set_xlabel(r'$\tau J/\hbar$')
    ax.set_ylabel(r'$\tilde \beta W_{ex} \ $ or $\ \ D$ or $\ \Delta \tilde S$')
    #ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
    #ax.set_xlim(1e-1, 1e3)
    #ax.set_ylim(top=1)
    ax.set_box_aspect(0.85)
    plt.savefig(f'D:\MESTRADO\spins3_1varia\Beta_star/beta_tilde,beta={beta},h_0={h_0},DELTA={DELTA}.png', bbox_inches='tight', dpi=600)
