import matplotlib.pyplot as plt
import numpy as np


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



taupoints = np.logspace(-1, 4, 20)
n_points = [8]
beta = 0.1
J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25
h_0 = hz0
DELTA = Delta_h


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
obs_entropy = np.real(np.loadtxt(f'Sobs_fine.txt', complex))
diag_entropy = np.real(np.loadtxt(f'Sd_points.txt', complex))
obs_entropy_traditional = np.real(np.loadtxt(f'Sobs_trad.txt', complex))
Sobs_coarse = np.real(np.loadtxt(f'Sobs_coarse.txt', complex))
Strad_coarse = np.real(np.loadtxt(f'Strad_coarse.txt', complex))

print(Strad_coarse)

"""
fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(obs_entropy)[0, :], 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints, np.real(diag_entropy)[0, :], 'rv', label=r'$\Delta S_{diag}$')
ax.loglog(taupoints, np.real(Dad_points)[0, :], 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(obs_entropy_traditional)[0, :], 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Sobs_coarse)[0, :], 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints, np.real(diag_entropy)[0, :], 'rv', label=r'$\Delta S_{diag}$')
ax.loglog(taupoints, np.real(Dad_points)[0, :], 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(Strad_coarse)[0, :], 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
# ax.set_title(r'Coarse-grained measurement', fontsize=15)
ax.set_box_aspect(0.85)
plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Sobs_coarse[0,:])/n_points[0], 'r.', label=f'$L = {n_points[0]}$')
ax.loglog(taupoints, np.real(Sobs_coarse[1,:])/n_points[1], 'k.', label=f'$L = {n_points[1]}$')
ax.loglog(taupoints, np.real(Sobs_coarse[2,:])/n_points[2], 'b.', label=f'$L = {n_points[2]}$')
#ax.loglog(taupoints, np.real(Dad_points[3,:]), 'g.', label=f'$L = {n_points[3]}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_ylabel(r'$S_{obs}/L$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(0.85)
#plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\coarse-grained entropy/DeltaS_obs_coarse beta=0,1 h0=0,1 DELTA=0,4.png', dpi=400, bbox_inches='tight')
plt.show()
"""

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(obs_entropy), 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints, np.real(diag_entropy), 'rv', label=r'$\Delta S_{d}$')
ax.loglog(taupoints, np.real(Dad_points), 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(obs_entropy_traditional), 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\coarse-grained entropy/spins8_fine_nonint_notrand_h0,1_DELTA0,4.png', bbox_inches='tight', dpi=600)


fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(taupoints, np.real(Sobs_coarse), 'k^', label=r'$\Delta S_{obs}$')
ax.loglog(taupoints, np.real(diag_entropy), 'rv', label=r'$\Delta S_{d}$')
ax.loglog(taupoints, np.real(Dad_points), 'bx', label=r'$D[\rho(\tau)||\rho_{ad}(\lambda_\tau)]$')
ax.loglog(taupoints, np.real(Strad_coarse), 'y.', label=r'$\Delta S_{obs, trad}$')
ax.legend(fancybox=False, edgecolor='black', loc='lower left', fontsize=14)
ax.set_xlabel(r'$\tau J/\hbar$')
# ax.set_title(r'Coarse-grained measurement', fontsize=15)
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\coarse-grained entropy/spins8_coarse_nonint_notrand_h0,1_DELTA0,4.png', bbox_inches='tight', dpi=600)