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


DeltaS_points = np.real(np.loadtxt(f'D_tilde.txt', complex))
DeltaS_star = np.real(np.loadtxt(f'D_star.txt', complex))
Dad_points = np.real(np.loadtxt(f'D_ad.txt', complex))
W_points = np.real(np.loadtxt(f'W.txt', complex))
Wad_points = np.real(np.loadtxt(f'W_ad.txt', complex))
Wex_points = W_points - Wad_points
Delta_Svol = np.real(np.loadtxt(f'DeltaS_vol.txt', complex))
sz_sum_avg = np.real(np.loadtxt(f'Sz_sum.txt', complex))
Wise_points = np.real(np.loadtxt(f'W_ise.txt', complex))
W_th = W_points - Wise_points
DeltaS_d = np.real(np.loadtxt(f'DeltaS_d.txt', complex))

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
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION\LZ quantifiers B0=-2.png', dpi=400, bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Dad_points - DeltaS_d, 'k.', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)] - \Delta S_d$')
ax.loglog(taupoints, dif_fit, 'k--', label=r'$\sim \tau^{-4}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION\LZ difference B0=-2.png', dpi=400, bbox_inches='tight')
#plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, Dad_points, 'b.', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, Wex_points, 'r.', label=r'$W_{ex}/J$')
ax.loglog(taupoints, Sd_fit, 'k--', label=r'$\sim \tau^{-2}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
plt.savefig(r'D:\MESTRADO\LZ_DISSERTATION\LZ Wex and A B0=-2.png', dpi=400, bbox_inches='tight')
#plt.show()