import numpy as np
import matplotlib.pyplot as plt

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


n_points = np.arange(3, 14)
tau_ad_max_list = np.real(np.loadtxt('tau_ad_max.txt', complex))
tau_ad_avg_list = np.real(np.loadtxt('tau_ad_avg.txt', complex))

fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy(n_points, np.real(tau_ad_max_list), 'k.', label=r'$\tau_{ad}$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\tau_{ad}J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\adiabatic_time_scale/max.png', dpi=400, bbox_inches='tight')

fig, ax = plt.subplots()
ax.grid(True)
ax.semilogy(n_points, np.real(tau_ad_avg_list), 'k.', label=r'$\tau_{ad}$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\tau_{ad}J/\hbar$')
ax.set_box_aspect(0.85)
plt.savefig(r'D:\MESTRADO\nonintegrable ising\not random\adiabatic_time_scale/avg.png', dpi=400, bbox_inches='tight')

