import numpy as np
import scipy.linalg as sl
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


L_list = np.logspace(0.4, 4, 40).astype(int)
DeltaS_list = []

J = 1
beta = 0.1
h_0 = 1.1
h_tau = 2.2

DELTA = h_tau - h_0



def lamda(k, h):
    return h + 2*J*np.cos(np.pi*k/(L+1))


def H(k, h):
    aux = np.zeros([2,2], complex)
    aux[0, 0] = lamda(k, h) - h/2
    aux[1, 1] = -h/2
    return aux


def Ef_thermal(b):
    ln_part_tau = 0
    aux = 0
    for k in k_list:
        ham_tau = H(k, h_tau)
        eigenvalues_tau = np.linalg.eigvalsh(ham_tau)
        probs_tau = np.exp(-b*eigenvalues_tau)
        ln_part_tau += np.log(np.sum(probs_tau))
        probs_tau /= np.sum(probs_tau)
        aux += np.dot(probs_tau, eigenvalues_tau)
    return aux, ln_part_tau


for L in L_list:
    k_list = np.arange(1, L+1)

    E_ad = 0
    E_0 = 0
    ln_part_0 = 0
    for k in k_list:
        eigenvalues_0 = np.array([lamda(k, h_0) - h_0/2, -h_0/2])
        probs_0 = np.exp(-beta*eigenvalues_0)
        s = np.sum(probs_0)
        ln_part_0 += np.log(s)
        probs_0 /= s
        E_0 += np.dot(probs_0, eigenvalues_0)
        eigenvalues_tau = np.array([lamda(k, h_tau) - h_tau/2, -h_tau/2])
        E_ad += np.dot(probs_0, eigenvalues_tau)

    def eq_to_find_beta_star(b):
        E_b, ln_part_b = Ef_thermal(b)
        return E_ad - E_b

    beta_star = fsolve(eq_to_find_beta_star, beta)[0]
    E_star, ln_part_star = Ef_thermal(beta_star)
    print(L, beta_star, E_ad, E_star, ln_part_star)
    DeltaS_star = beta_star*E_star + ln_part_star - beta*E_0 - ln_part_0
    DeltaS_list.append(DeltaS_star)

DeltaS_list = np.array(DeltaS_list)
L_list = np.array(L_list)

print(L_list)
print(DeltaS_list)

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

#plt.rcParams['axes.autolimit_mode'] = 'data'
plt.rc('text', usetex=False)

fig, ax = plt.subplots()
ax.grid(True)
ax.loglog(L_list, np.real(DeltaS_list), 'k.', label=r'$\Delta S^*$')
ax.legend(fancybox=False, edgecolor='black', loc='lower right', fontsize=14)
ax.set_xlabel(r'Number of spins')
ax.set_box_aspect(0.85)
plt.savefig('integrable Ising large chain', dpi=200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.grid(True)
ax.semilogx(L_list, np.real(DeltaS_list)/L_list, 'k.', label=r'$\Delta S^*/L$')
ax.legend(fancybox=False, edgecolor='black', loc='lower right', fontsize=14)
ax.set_xlabel(r'Number of spins')
ax.set_box_aspect(0.85)
plt.savefig('integrable Ising large chain divided by L', dpi=400, bbox_inches='tight')
plt.show()
