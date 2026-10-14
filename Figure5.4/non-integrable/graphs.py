import matplotlib.pyplot as plt
import numpy as np


beta = 0.1

J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25

h_0 = hz0
DELTA = Delta_h

n_points = np.arange(3, 15)

D_points = np.loadtxt('DeltaS_star', complex)
D_points_2 = np.loadtxt('D2', complex)
Delta_min = np.loadtxt('Delta E_min', complex)


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
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#plt.rcParams['axes.autolimit_mode'] = 'data'
plt.rc('text', usetex=True)


path = r'D:\MESTRADO\nonintegrable ising\not random\exact_diag_beta'+str(beta)
path2 = r'D:\MESTRADO\nonintegrable ising\not random\exact_diag_beta'+str(beta)+'_defense'


fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy([int(n) for n in n_points], np.real(D_points), 'k^', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\beta^*}(\lambda_\tau)]$')
ax.semilogy([int(n) for n in n_points], np.real(D_points_2), 'rv', label=r'$D[\Pi_{\beta^*}(\lambda_\tau)||\rho_{ad}(\lambda_\tau)]$')
#ax.loglog([2*int(n) for n in n_points], D_fit1, 'r-', label=r'Fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path}/nonintegrable not random ising exact diagonalization', bbox_inches='tight', dpi=100)
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.plot([int(n) for n in n_points], np.real(D_points), 'k^', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\beta^*}(\lambda_\tau)]$')
ax.plot([int(n) for n in n_points], np.real(D_points_2), 'rv', label=r'$D[\Pi_{\beta^*}(\lambda_\tau)||\rho_{ad}(\lambda_\tau)]$')
#ax.loglog([2*int(n) for n in n_points], D_fit1, 'r-', label=r'Fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path}/nonintegrable not random ising exact diagonalization LINEAR', bbox_inches='tight', dpi=400)
#plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy([int(n) for n in n_points], np.real(D_points)/np.array([int(n) for n in n_points]), 'k^', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\beta^*}(\lambda_\tau)]/L$')
ax.semilogy([int(n) for n in n_points], np.real(D_points_2)/np.array([int(n) for n in n_points]), 'rv', label=r'$D[\Pi_{\beta^*}(\lambda_\tau)||\rho_{ad}(\lambda_\tau)]/L$')
#ax.loglog([2*int(n) for n in n_points], D_fit1, 'r-', label=r'Fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins ($L$)')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path}/nonintegrable not random ising exact diagonalization divided by N', bbox_inches='tight', dpi=100)
#plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.plot([int(n) for n in n_points], np.real(D_points)/np.array([int(n) for n in n_points]), 'k^', label=r'$D[\rho_{ad}(\lambda_\tau)||\Pi_{\beta^*}(\lambda_\tau)]/L$')
ax.plot([int(n) for n in n_points], np.real(D_points_2)/np.array([int(n) for n in n_points]), 'rv', label=r'$D[\Pi_{\beta^*}(\lambda_\tau)||\rho_{ad}(\lambda_\tau)]/L$')
#ax.loglog([2*int(n) for n in n_points], D_fit1, 'r-', label=r'Fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins ($L$)')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path}/nonintegrable not random ising exact diagonalization divided by N LINEAR', bbox_inches='tight', dpi=400)
#plt.show()


fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy([int(n) for n in n_points], np.real(Delta_min), 'r.', label=r'$\Delta E_{min}$')
#ax.loglog([2*int(n) for n in n_points], D_fit1, 'r-', label=r'Fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path}/nonintegrable not random ising exact diagonalization Delta E_min', bbox_inches='tight', dpi=100)
#plt.show()



##### DEFENSE #####

fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy([int(n) for n in n_points], np.real(D_points)/np.array([int(n) for n in n_points]), 'k.', label=r'$\Delta S^*/L$')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins ($L$)')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path2}/nonintegrable not random ising exact diagonalization divided by N', bbox_inches='tight', dpi=200)
#plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy([int(n) for n in n_points], np.real(Delta_min)/np.array([int(n) for n in n_points]), 'k.', label=r'$(W_{ad} - W_{ise})/L$')
ax.legend(fancybox=False, edgecolor='black', loc='upper right', fontsize=14)
ax.set_xlabel(r'Number of spins ($L$)')
ax.set_ylabel(r'$ W/L $')
#ax.set_title(r'Comparison between $\beta^*W_{ex}(\tau)$ and $D[\rho(\tau)||\Pi_{\beta^*}(\lambda_\tau)]$', fontsize=15)
#ax.set_ylim(2e-7, 5e-5)
ax.set_box_aspect(0.85)
plt.savefig(f'{path2}/Wad - Wise nonintegrable not random ising exact diagonalization divided by N', bbox_inches='tight', dpi=200)
#plt.show()
