import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
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


L = 4
beta = 1.0
h_0, Delta_h = 0.1, 2.0


ln_ratios = np.real(np.loadtxt(f'ln_ratios Y 4 spins, beta=1.0, h0=0.1. DELTA=2.0', complex))
Y_line_points = np.real(np.loadtxt(f'Y for 4 spins, beta=1.0, h0=0.1. DELTA=2.0', complex))

fit, cov = np.polyfit(Y_line_points, ln_ratios, deg=1, cov=True)
print(f'Linear fit...')
print(f'a = {ufloat(fit[0], np.sqrt(cov[0, 0]))}')
print(f'b = {ufloat(fit[1], np.sqrt(cov[1, 1]))}')

ln_ratios_fit = fit[0]*Y_line_points + fit[1]


fig, ax = plt.subplots()
ax.grid(True)
ax.plot(Y_line_points, ln_ratios, 'k.', label='Numerical data')
ax.plot(Y_line_points, ln_ratios_fit, 'r-', label='Linear fit')
ax.legend(fancybox=False, edgecolor='black', loc='upper left', fontsize=16)
ax.set_xlabel(r'$\overline Y$')
ax.set_ylabel(r'$\ln [p(\overline Y)/\tilde p(-\overline Y)]$')
#ax.set_xlim(1e-1, 1e3)
#ax.set_ylim(top=1)
ax.set_box_aspect(1)
plt.savefig(f'D:\MESTRADO\PAPER/Y_DFT 4 spins nonintegrable random beta={beta} Deltah={Delta_h}.png', bbox_inches='tight', dpi=600)
