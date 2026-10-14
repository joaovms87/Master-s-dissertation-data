######### NONINTEGRABLE ##########

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from numpy.random import normal
from scipy.optimize import fsolve
from scipy.stats import entropy


beta = 0.1

J = 1.0
hx = 1.1
hz0, Delta_h = 0.1, 0.4
h1, hL = 0.25, -0.25

h_0 = hz0
DELTA = Delta_h

#imag_dt = 0.02  # for imaginary time evolution
#rng = np.random.default_rng(42)


tau = 1

Id = np.array([[1., 0.],
               [0., 1.]])
sigma_x = np.array([[0., 1.],
                   [1., 0.]])
sigma_z = np.array([[1., 0.],
                   [0., -1.]])

def tensor(matrices):
    tens = np.kron(matrices[0], matrices[1])
    for k in range(2, len(matrices)):
        tens = np.kron(tens, matrices[k])
    return tens


def h(t):  # protocol for gamma1 variation
    if t==0:
        return h_0
    else:
        return h_0 + DELTA/tau*t


def Ham(t):
    h_list = np.ones(n_spins, float)*h(t)

    s_z = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_z
        s_z += h_list[j]*tensor(matrix_list)

    s_x = np.zeros([2 ** n_spins, 2 ** n_spins], float)
    for j in range(n_spins):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j] = sigma_x
        s_x += hx_list[j] * tensor(matrix_list)

    s_z_int = np.zeros([2**n_spins, 2**n_spins], float)
    for j in range(n_spins-1):
        matrix_list = [Id for y in range(n_spins)]
        matrix_list[j], matrix_list[j+1] = sigma_z, sigma_z
        s_z_int += J_list[j]*tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[0] = sigma_z
    term1 = h1 * tensor(matrix_list)

    matrix_list = [Id for y in range(n_spins)]
    matrix_list[-1] = sigma_z
    termL = hL * tensor(matrix_list)

    return s_z + s_x + s_z_int + term1 + termL

def rho_eq_diag(b):
  aux = np.exp(-b*values_tau)
  return aux/sum(aux)


def equation_ad(b):
    return np.dot((rho_ad_diag - rho_eq_diag(b)), values_tau)


def equation_line(b):
    return entropy(rho_ad_diag) - entropy(rho_eq_diag(b))



n_points = np.arange(3, 15)


D_points = np.empty(len(n_points), complex)
D_points_2 = np.empty(len(n_points), complex)
D_random = np.empty(len(n_points), complex)
Delta_min = np.empty(len(n_points), complex)
#Delta_S = np.empty(len(n_points), complex)


for q, n in enumerate(n_points):
  n_spins = int(n)
  print(n_spins)
  w = 2**n_spins
  L = n_spins

  J_list = np.ones(n)
  hx_list = np.ones(n)*hx
  """
  J_list = rng.normal(J_mean, J_std, L - 1)
  hx_list = rng.normal(hx_mean, hx_std, L)
  """

  H_0 = Ham(0)
  values0 = sl.eigvalsh(H_0)

  H_tau = Ham(tau)
  values_tau = sl.eigvalsh(H_tau)

  rho_ad_diag = np.exp(-beta*values0)
  rho_ad_diag /= np.sum(rho_ad_diag)

  beta_star = fsolve(equation_ad, beta)[0]
  D_points[q] = entropy(rho_ad_diag, rho_eq_diag(beta_star))
  D_points_2[q] = entropy(rho_eq_diag(beta_star), rho_ad_diag)

  beta_line = fsolve(equation_line, beta)[0]
  Delta_min[q] = np.dot(rho_ad_diag, values_tau) - np.dot(rho_eq_diag(beta_line), values_tau)



np.savetxt('DeltaS_star', D_points)
np.savetxt('D2', D_points_2)
np.savetxt('Delta E_min', Delta_min)
