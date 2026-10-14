import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
from uncertainties import ufloat

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

n_points = [3]

DeltaS_points = np.real(np.loadtxt(f'D_tilde.txt', complex))
DeltaS_star = np.real(np.loadtxt(f'D_star.txt', complex))
Dad_points = np.real(np.loadtxt(f'D_ad.txt', complex))
W_points = np.real(np.loadtxt(f'W.txt', complex))
Wad_points = np.real(np.loadtxt(f'W_ad.txt', complex))
Wex_points = W_points - Wad_points
Delta_Svol = np.real(np.loadtxt(f'DeltaS_vol.txt', complex))
sz_sum_avg = np.real(np.loadtxt(f'Sz_sum.txt', complex))
DeltaE_line = np.real(np.loadtxt(f'DeltaE_line.txt', complex))
Wise_points = np.real(np.loadtxt(f'W_ise.txt', complex))
W_th = W_points - Wise_points
DeltaS_d = np.real(np.loadtxt(f'DeltaS_d.txt', complex))
beta_tilde = np.real(np.loadtxt(f'beta_tilde.txt', complex))
beta_star = np.real(np.loadtxt(f'beta_star.txt', complex))
beta_line = np.real(np.loadtxt(f'beta_line.txt', complex))

D_th = np.real(np.loadtxt(f'D_th.txt', complex))
D_ad_tilde = np.real(np.loadtxt(f'D_ad_tilde.txt', complex))
D_rho_star = np.real(np.loadtxt(f'D_rho_star.txt', complex))


taupoints = np.logspace(-1, 3, len(Delta_Svol))

for i, tau in enumerate(taupoints):
    if tau > 50:
        break


fit, cov = np.polyfit(np.log(taupoints[i:]), np.log(Delta_Svol[i:]), deg=1, cov=True)
print(f'DeltaS_vol fit...')
print(ufloat(fit[0], np.sqrt(cov[0, 0])))
Svol_fit = taupoints**(-2) * np.exp(fit[1])

fit, cov = np.polyfit(np.log(taupoints[i:]), np.log(DeltaS_d[i:]), deg=1, cov=True)
print(f'DeltaS_d fit...')
print(ufloat(fit[0], np.sqrt(cov[0, 0])))
Sd_fit = taupoints**(-2) * np.exp(fit[1])

fit, cov = np.polyfit(np.log(taupoints[i:]), np.log(Dad_points[i:] - DeltaS_d[i:]), deg=1, cov=True)
print(f'D_ad - DeltaS_d fit...')
print(ufloat(fit[0], np.sqrt(cov[0, 0])))
dif_fit = taupoints**(-4) * np.exp(fit[1])




fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Delta_Svol, 'r.', label=r'$\Delta S_{vol}$')
ax.loglog(taupoints, Dad_points, 'b.', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, DeltaS_d, 'kx', label=r'$\Delta S_d$')

ax.loglog(taupoints, Svol_fit, 'r--', label=r'$\sim \tau^{-2}$')
#ax.loglog(taupoints, Sd_fit, 'k--', label=r'$\sim \tau^{-2}$')

ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\quantifiers h0=1,1 DELTA=1 beta=1.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Dad_points - DeltaS_d, 'k.', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)] - \Delta S_d$')
ax.loglog(taupoints, dif_fit, 'k--', label=r'$\sim \tau^{-4}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\difference h0=1,1 DELTA=1 beta=1.png', dpi=400, bbox_inches='tight')
#plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Dad_points, 'b.', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, Wex_points, 'r.', label=r'$W_{ex}/J$')
ax.loglog(taupoints, Sd_fit, 'k--', label=r'$\sim \tau^{-2}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\Wex and A h0=1,1 DELTA=1 beta=1.png', dpi=400, bbox_inches='tight')
#plt.show()

# plot D_star and Delta_S in log scale together with Wex
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(beta_star*Wex_points), 'rv', label=r'$\beta^*W_{ex}(\tau)$')
ax.loglog(taupoints, np.real(DeltaS_star), 'k--', label=r'$\Delta S^*$')
ax.loglog(taupoints, np.real(D_rho_star - DeltaS_star), 'k^', label=r'$D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]-\Delta S^*$')
ax.loglog(taupoints, np.real(D_rho_star), 'b.', label=r'$D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
#ax.set_ylabel(r'$\beta^*W_{ex} \ $ or $\ \ D$ or $\ \Delta S^*$')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\beta_star h0=1,1 DELTA=1 beta=1.png', bbox_inches='tight', dpi=400)




# plot D_tilde and Delta_S_tilde in log scale together with Wex
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(beta_tilde*Wex_points), 'rv', label=r'$\tilde \beta W_{ex}(\tau)$')
ax.loglog(taupoints, np.real(D_ad_tilde), 'k--', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\tilde \beta}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(DeltaS_points - D_ad_tilde), 'k^', label=r'$\Delta \tilde S - D[\rho_{ad}(\lambda_\tau)||\Pi_{\tilde \beta}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(DeltaS_points), 'b.', label=r'$\Delta \tilde S$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
#ax.set_ylabel(r'$\tilde \beta W_{ex} \ $ or $\ \ D$ or $\ \Delta \tilde S$')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\beta_tilde h0=1,1 DELTA=1 beta=1.png', bbox_inches='tight', dpi=400)



fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(beta_line*Wex_points), 'rv', label=r'$\overline \beta W_{ex}(\tau)$')
ax.loglog(taupoints, np.real(beta_line*DeltaE_line), 'k--', label=r'$\overline \beta \Delta \overline E$')
ax.loglog(taupoints, np.real(D_th - beta_line*DeltaE_line), 'k^', label=r'$D[\rho(\tau)||\Pi_{\overline \beta}(\lambda_\tau)] - \overline \beta \Delta \overline E$')
ax.loglog(taupoints, np.real(D_th), 'b.', label=r'$D[\rho(\tau)||\Pi_{\overline \beta}(\lambda_\tau)] = \overline \beta W_{ex}^{th}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
#ax.set_ylabel(r'$\tilde \beta W_{ex} \ $ or $\ \ D$ or $\ \Delta \tilde S$')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\beta_line h0=1,1 DELTA=1 beta=1.png', bbox_inches='tight', dpi=400)




#### DEFENSE ####

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Dad_points, 'bv', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, beta_star*Wex_points, 'r^', label=r'$\beta^* W_{ex}$')
#ax.loglog(taupoints, beta_line* W_th, 'g.', label=r'$\overline \beta W_{ex}^{th}$')
ax.loglog(taupoints, DeltaS_d, 'kx', label=r'$\Delta S_d$')
#ax.loglog(taupoints, DeltaS_points, 'k+', label=r'$\Delta \tilde S$')
ax.loglog(taupoints, Sd_fit, 'k--', label=r'$\sim \tau^{-2}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\integrable ising\spins3_1varia\defense h0=1,1 DELTA=1 beta=1.png', dpi=400, bbox_inches='tight')





fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, abs(Dad_points - beta_star*Wex_points), 'k')
plt.show()
