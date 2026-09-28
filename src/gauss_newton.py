import numpy as np
import geo_helper as gh
from geo_helper import pvData
import test_utils as tu

# def solve_gauss_newton_translation(lats, lons, v_e_obs, v_n_obs, euler_pole):
#   return solve_gauss_newton_translation(lats, lons, v_e_obs, v_n_obs, None, None, euler_pole)


def solve_gauss_newton_translation(pvData, euler_pole):
  """
  Computes the true average translation offset vector over a regional network.
  Unified to handle standard deg/Ma and mm/yr inputs over a metric radius.
  """
  num_stations = len(pvData.lats)

  x = {'t_x': 0, 't_y': 0, 's': 0, 'r': 0}

  if num_stations < 4:  # need at least 4 points to solve
    return np.zeros(2)

  # 3D ECEF parameter tracking vector [Tx, Ty, Tz]
  T = np.zeros(3)

  # Convert deg/Ma directly to rad/Ma
  omega_rad_ma = np.radians(euler_pole.omega)
  pole_phi = np.radians(euler_pole.lat)
  pole_lam = np.radians(euler_pole.long)

  # Build 3D rotation vector O (rad/Ma)
  O = omega_rad_ma * np.array([
      np.cos(pole_phi) * np.cos(pole_lam),
      np.cos(pole_phi) * np.sin(pole_lam),
      np.sin(pole_phi)
  ])

  for iteration in range(3):
    J = np.zeros((2 * num_stations, 3))
    r = np.zeros(2 * num_stations)

    for i in range(num_stations):
      phi = np.radians(pvData.lats[i])
      lam = np.radians(pvData.longs[i])

      sw_e = 1.0 / pvData.s_es[i] if pvData.s_es is not None else 1.0
      sw_n = 1.0 / pvData.s_ns[i] if pvData.s_ns is not None else 1.0

      e_hat = np.array([-np.sin(lam), np.cos(lam), 0.0])
      n_hat = np.array([-np.sin(phi) * np.cos(lam), -
                       np.sin(phi) * np.sin(lam), np.cos(phi)])

      # 1. Build position vector in km
      P_m = gh.R * np.array([np.cos(phi) * np.cos(lam),
                            np.cos(phi) * np.sin(lam), np.sin(phi)])

      # 2. Rigid rotation velocity in meters per million years (m/Ma)
      V_m_ma = np.cross(O, P_m)

      # 3. Down-scale from m/Ma to mm/yr (1 m/Ma = 1e-3 mm/yr)
      # This step cleanly scales the velocity framework without altering the position radius
      v_e_rot = np.dot(V_m_ma, e_hat) * 1e-3
      v_n_rot = np.dot(V_m_ma, n_hat) * 1e-3

      # 4. Model tracking for current translations
      v_e_trans = np.dot(T, e_hat)
      v_n_trans = np.dot(T, n_hat)

      v_e_pred = v_e_rot + v_e_trans
      v_n_pred = v_n_rot + v_n_trans

      idx_e = 2 * i
      idx_n = 2 * i + 1

      # Residual matching
      r[idx_e] = (pvData.v_es[i] - v_e_pred) * sw_e
      r[idx_n] = (pvData.v_ns[i] - v_n_pred) * sw_n

      J[idx_e, 0:3] = e_hat * sw_e
      J[idx_n, 0:3] = n_hat * sw_n

    try:
      delta, residual, _, _ = np.linalg.lstsq(J, r, rcond=None)
    except np.linalg.LinAlgError:
      break

    tu.test_regression_stats(delta, J, r, residual, False)

    T += delta
    if np.linalg.norm(delta) < 1e-6:
      break

  # Project 3D vector to local horizontal coordinates at dataset geometric center
  center_phi = np.radians(np.mean(pvData.lats))
  center_lam = np.radians(np.mean(pvData.longs))

  c_e_hat = np.array([-np.sin(center_lam), np.cos(center_lam), 0.0])
  c_n_hat = np.array([-np.sin(center_phi) * np.cos(center_lam), -
                     np.sin(center_phi) * np.sin(center_lam), np.cos(center_phi)])

  t_east_avg = np.dot(T, c_e_hat)
  t_north_avg = np.dot(T, c_n_hat)

  return np.array([t_east_avg, t_north_avg])


# Gauss-Newton 2d solver for translation, rotation and scale in 2D
# meters and mm/Y units
def solve_gauss_newton_2D_transform_geo(testPvData, center_ploc, normalize=True):
  sample_es, sample_ns = gh.getSamplePoints(testPvData.longs, testPvData.lats, center_ploc)
  return solve_gauss_newton_2D_transform(sample_es, sample_ns, testPvData.v_es, testPvData.v_ns, normalize)

# lats and longs should be normalized relative to "center" of rotation for best results
# meters and mm/Y units
# meters and mm/Y units
def solve_gauss_newton_2D_transform(sample_e, sample_n, v_e, v_n, normalize=True, ):
  N = len(v_e)

  if N < 4:  # need at least 4 points to solve
    return np.zeros(2)

  c = [0.0, 0.0]
  if normalize:
    c[0] = np.mean(sample_e)
    c[1] = np.mean(sample_n)

  J = np.zeros((2 * N, 4))
  R = np.zeros(2 * N)

  j_idx = 0
  for i in range(N):
    u = sample_e[i] - c[0]
    v = sample_n[i] - c[1]

    # Calculate Jacobian elements for Dx
    J[j_idx, 0] = 1.0
    J[j_idx, 1] = 0.0
    J[j_idx, 2] = u
    J[j_idx, 3] = v

    R[j_idx] = v_e[i]
    j_idx += 1

    J[j_idx, 0] = 0.0
    J[j_idx, 1] = 1.0
    J[j_idx, 2] = v
    J[j_idx, 3] = -u

    R[j_idx] = v_n[i]
    j_idx += 1

    # JT = J.T
    # JTJ = JT.dot(J)
    # JTR = JT.dot(R)

  x, residuals, rank, s = np.linalg.lstsq(J, R, rcond=None)
  tu.test_regression_stats(x, J, R, residuals, False)

  # return  {'t_x' : x[0], 't_y': x[1], 's' : x[2], 'r' : x[3]}
  return np.array([x[0], x[1]])


# Gauss-Newton 2d *weighted* solver for translation, rotation and scale in 2D
# meters and mm/Y units
def solve_gauss_newton_2D_transform_geo_wtd(pvData, center_ploc, normalize=True):
  sample_e, sample_n = gh.getSamplePoints(pvData.longs, pvData.lats, center_ploc)
  return solve_gauss_newton_2D_transform_wtd(
    sample_e, sample_n, pvData.v_es, pvData.v_ns, pvData.s_es, pvData.s_ns, normalize)


# lats and longs should be normalized relative to "center" of rotation for best results
# meters and mm/Y units
def solve_gauss_newton_2D_transform_wtd(sample_e, sample_n, v_e, v_n, w_e, w_n, normalize=True, ):
  N = len(v_e)
  x = {'t_x': 0, 't_y': 0, 's': 0, 'r': 0}
  if N < 4:  # need at least 4 points to solve
    return x

  c = [0.0, 0.0]
  if normalize:
    c[0] = np.mean(sample_e)
    c[1] = np.mean(sample_n)

  J = np.zeros((2 * N, 4))
  R = np.zeros(2 * N)

  j_idx = 0
  for i in range(N):
    u = sample_e[i] - c[0]
    v = sample_n[i] - c[1]

    sw_e = 1.0 / w_e[i] if w_e is not None else 1.0
    sw_n = 1.0 / w_n[i] if w_n is not None else 1.0

    # Weighted Jacobian and Residual elements for Easting (Dx)
    J[j_idx, 0] = 1.0 * sw_e
    J[j_idx, 1] = 0.0 * sw_e
    J[j_idx, 2] = u * sw_e
    J[j_idx, 3] = v * sw_e

    R[j_idx] = v_e[i] * sw_e
    j_idx += 1

    # Weighted Jacobian and Residual elements for Northing (Dy)
    J[j_idx, 0] = 0.0 * sw_n
    J[j_idx, 1] = 1.0 * sw_n
    J[j_idx, 2] = v * sw_n
    J[j_idx, 3] = -u * sw_n

    R[j_idx] = v_n[i] * sw_n
    j_idx += 1

  try:
    # np.linalg.lstsq solves the system: (sqrt(W)*J)^T * (sqrt(W)*J) * x = (sqrt(W)*J)^T * (sqrt(W)*r)
    # Which simplifies exactly to: J^T * W * J * x = J^T * W * r
    x, residuals, rank, s = np.linalg.lstsq(J, R, rcond=None)
    tu.test_regression_stats(x, J, R, residuals, False)
  except Exception as e:
    print(f"solve_gauss_newton_2D_transform_wtd exception: \n{str(e)}")
    x = ([0], [0])

  # return {'t_x' : x[0], 't_y': x[1], 's' : x[2], 'r' : np.degrees(x[3])}
  return np.array([x[0], x[1]])

# Gauss-Newton 2d solver for translation, rotation and scale in 2D
# meters and mm/Y units
def getAverageValocity_geo(sample_long, sample_lat, v_e, v_n, euler_pole):
  sample_e, sample_n = gh.getSamplePoints(sample_long, sample_lat, euler_pole)
  return getWeightedAverageValocity(sample_e, sample_n, v_e, v_n)

# lats and longs should be normalized relative to "center" of rotation for best results

# meters and mm/Y units
def getWeightedAverageValocity(sample_e, sample_n, v_e, v_n):
  N = len(v_e)

  J = np.zeros((2 * N, 2))
  R = np.zeros(2 * N)

  j_idx = 0
  for i in range(N):

    J[j_idx, 0] = 1.0
    J[j_idx, 1] = 0.0

    R[j_idx] = v_e[i]
    j_idx += 1

    J[j_idx, 0] = 0.0
    J[j_idx, 1] = 1.0

    R[j_idx] = v_n[i]
    j_idx += 1

  x, residuals, rank, s = np.linalg.lstsq(J, R, rcond=None)
  tu.test_regression_stats(x, J, R, residuals, False)

  return np.array([x[0], x[1]])

def gn_print(x):
  print(f"\nt_x:\t {x['t_x']:.5f}")
  print(f"t_y:\t {x['t_y']:.5f}")
  print(f"s:  \t {x['s']:.5f}")
  print(f"r:  \t {x['r']:.5f}°")

def solve_gauss_newton_transform(pvData, raw_pole, useStereo):
  # apply Gauss-Newton analysis to get any translation (non-rotation) components
  if useStereo:
    offset = solve_gauss_newton_2D_transform_geo_wtd(pvData, raw_pole.ploc())
    # print(f"offset (Stereo): {offset}")
  else:
    offset = solve_gauss_newton_translation(pvData, raw_pole)
    # print(f"offset (Gemini): {offset}")
  return offset
