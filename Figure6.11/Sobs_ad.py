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
  aux = np.exp(-b*evals)
  return aux/sum(aux)


def equation_ad(b):
    return np.dot((rho_ad_diag - rho_eq_diag(b)), evals)


def rho_eq_tau(beta, evals, evecs):

    rho = evecs @ np.diag(np.exp(-beta * evals)) @ evecs.conj().T

    return rho / np.trace(rho)




n_points = np.arange(3, 13)



#Delta_S = np.empty(len(n_points), complex)
S_obs_ad = np.empty(len(n_points), complex)


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

  H0 = Ham(0)
  eigvals0, eigvecs0 = sl.eigh(H0)

  rho0 = eigvecs0 @ np.diag(np.exp(-beta * eigvals0)) @ eigvecs0.conj().T

  Hf = Ham(tau)
  evals, evecs = sl.eigh(Hf)

  rho_ad_diag = np.exp(-beta*eigvals0)
  rho_ad_diag /= np.sum(rho_ad_diag)

  rho_ad = evecs @ np.diag(rho_ad_diag) @ evecs.conj().T

  beta_star = fsolve(equation_ad, beta)[0]

  # COARSE GRAINING
  n_measurements = 10

  Mx_list = []
  Mx_list0 = []

  edges = np.linspace(evals[0], evals[-1], n_measurements + 1)
  for i in range(n_measurements):

      if i < n_measurements - 1:
          mask = (evals >= edges[i]) & (evals < edges[i + 1])
      else:
          mask = (evals >= edges[i]) & (evals <= edges[i + 1])

      M = np.zeros_like(Hf)

      for k in np.where(mask)[0]:
          M += np.outer(evecs[:, k], evecs[:, k].conj())

      Mx_list.append(M)

  edges = np.linspace(eigvals0[0], eigvals0[-1], n_measurements + 1)
  for i in range(n_measurements):

      if i < n_measurements - 1:
          mask = (eigvals0 >= edges[i]) & (eigvals0 < edges[i + 1])
      else:
          mask = (eigvals0 >= edges[i]) & (eigvals0 <= edges[i + 1])

      M = np.zeros_like(H0)

      for k in np.where(mask)[0]:
          M += np.outer(eigvecs0[:, k], eigvecs0[:, k].conj())

      Mx_list0.append(M)

  """
  Id_check = np.sum(Mx_list, axis=0)
  print(np.linalg.norm(Id_check - np.eye(len(evals))))

  Id_check = np.sum(Mx_list0, axis=0)
  print(np.linalg.norm(Id_check - np.eye(len(evals))))
  """

  coarse_p_rho = []
  coarse_p_prior = []
  coarse_p0 = []
  coarse_Id = []
  coarse_Id0 = []
  for i in range(n_measurements):
      Mx = Mx_list[i]
      Mx_0 = Mx_list0[i]
      coarse_p_rho.append(np.trace(Mx @ rho_ad))
      coarse_p_prior.append(np.trace(Mx @ rho_eq_tau(beta_star, evals, evecs)))
      coarse_p0.append(np.trace(Mx_0 @ rho0))

      coarse_Id.append(np.trace(Mx))
      coarse_Id0.append(np.trace(Mx_0))

  coarse_p_rho = np.real(np.array(coarse_p_rho))
  coarse_p_prior = np.real(np.array(coarse_p_prior))
  coarse_p0 = np.real(np.array(coarse_p0))

  p_tilde = np.real(np.exp(-beta_star * evals))
  p_tilde = p_tilde / sum(p_tilde)
  p_rho = np.real(np.diag(evecs.conj().T @ rho_ad @ evecs))

  p_0 = np.real(np.exp(-beta * eigvals0))

  coarse_Id = np.array(coarse_Id, dtype=float)
  coarse_Id0 = np.array(coarse_Id0, dtype=float)

  """
  print(type(coarse_Id))
  print(type(coarse_Id0))

  print(np.array(coarse_Id).dtype)
  print(np.array(coarse_Id0).dtype)

  print(np.array(coarse_p_rho).dtype)
  print(np.array(coarse_p0).dtype)
  """

  obs_entropy = entropy(p_tilde) - entropy(p_rho, p_tilde) - entropy(p_0)

  diag_entropy = entropy(p_rho) - entropy(p_0)
  print(diag_entropy)

  p_ones = np.ones(len(p_rho)) / len(p_rho)
  obs_entropy_traditional = - entropy(p_rho, p_ones) + entropy(p_0, p_ones)

  Sobs_coarse = entropy(p_tilde) - entropy(coarse_p_rho, coarse_p_prior) - entropy(p_0)
  Strad_coarse = - entropy(coarse_p_rho, coarse_Id) + entropy(coarse_p0, coarse_Id0)


  S_obs_ad[q] = Sobs_coarse

np.savetxt('S_obs_ad.txt', S_obs_ad)
