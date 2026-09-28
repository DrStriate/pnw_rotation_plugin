import numpy as np
from dataclasses import replace
from geo_helper import EulerPole, PLoc, pvData, getPAvel, R
import gauss_newton as gn
import test_utils as tu

def fit_euler_pole_linear(pvData):
  """
  Finds the exact best-fitting Euler pole using linear least squares.

  Args:
    pvData.lat (list/array): Latitudes of stations in decimal degrees
    pvData.lons (list/array): Longitudes of stations in decimal degrees
    v_east (list/array): East velocity components in mm/yr
    v_north (list/array): North velocity components in mm/yr

    align_pole tests for euler pole pointing to incoming n/s hemisphere (i.e Omega pole flip)
  """

  num_stations = len(pvData.lats)

  if num_stations < 3:
    return EulerPole(0, 0, 0)

  # Initialize design matrix A and observation vector B
  A = np.zeros((2 * num_stations, 3))
  B = np.zeros(2 * num_stations)

  sum_lats = 0
  for i in range(num_stations):
    # Convert input coordinates to radians
    phi = np.radians(pvData.lats[i])
    lam = np.radians(pvData.longs[i])
    sum_lats += pvData.lats[i]

    # weights for this station
    sw_e = 1.0 / pvData.s_es[i] if pvData.s_es is not None else 1.0
    sw_n = 1.0 / pvData.s_ns[i] if pvData.s_ns is not None else 1.0

    # Weighted East velocity row equations (even rows: 2*i)
    A[2*i, 0] = -R * np.sin(phi) * np.cos(lam) * sw_e
    A[2*i, 1] = -R * np.sin(phi) * np.sin(lam) * sw_e
    A[2*i, 2] = R * np.cos(phi) * sw_e
    B[2*i] = pvData.v_es[i] * sw_e

    # Weighted North velocity row equations (odd rows: 2*i+1)
    A[2*i+1, 0] = R * np.sin(lam) * sw_n
    A[2*i+1, 1] = -R * np.cos(lam) * sw_n
    A[2*i+1, 2] = 0.0 * sw_n
    B[2*i+1] = pvData.v_ns[i] * sw_n

  # Solves the weighted normal equations: A^T * W * A * omega = A^T * W * B
  omega_3D, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
  tu.test_regression_stats(omega_3D, A, B, residuals, False)

  # wx, wy, wz = omega_3D
  north_hemisphere = (sum_lats > 0.0)
  if (omega_3D[2] > 0) != north_hemisphere:  # if w and incoming data not in the same N/S hemisphere
    omega_3D = -omega_3D # flip pole

  # Convert the Cartesian angular velocity vector back into Euler Pole parameters
  # 1. Total angular rotation magnitude (rad/yr converted back to deg/Myr)
  omega_mag_rad = np.linalg.norm(omega_3D)

  # 2. Latitude and Longitude of the Pole
  lat_pole = np.degrees(np.arcsin(omega_3D[2] / omega_mag_rad))
  lon_pole = np.degrees(np.arctan2(omega_3D[1], omega_3D[0]))
  omega_deg_myr = np.degrees(omega_mag_rad)

  return EulerPole(lon_pole, lat_pole, omega_deg_myr, is_clockwise=True)


def fit_euler_pole_linear2(pvData):
  """
  Finds the best-fitting Euler pole and localized horizontal translation
  simultaneously, accounting for legacy m/Ma inputs safely.
  """

  num_stations = len(pvData.lats)

  A_joint = np.zeros((2 * num_stations, 5))
  B = np.zeros(2 * num_stations)

  sum_lats = 0

  for i in range(num_stations):
    phi = np.radians(pvData.lats[i])
    lam = np.radians(pvData.longs[i])
    sum_lats += pvData.lats[i]

    # weights for this station
    sw_e = 1.0 / pvData.s_es[i] if pvData.s_es is not None else 1.0
    sw_n = 1.0 / pvData.s_ns[i] if pvData.s_ns is not None else 1.0

    e_hat = np.array([-np.sin(lam), np.cos(lam), 0.0])
    n_hat = np.array([-np.sin(phi) * np.cos(lam), -
                     np.sin(phi) * np.sin(lam), np.cos(phi)])

    # 1. Build position vector 
    P = R * np.array([np.cos(phi) * np.cos(lam), np.cos(phi)
                     * np.sin(lam), np.sin(phi)])

    row_east_pole = np.cross(P, e_hat)
    row_north_pole = np.cross(P, n_hat)

    idx_e = 2 * i
    idx_n = 2 * i + 1

    A_joint[idx_e, 0:3] = row_east_pole * sw_e
    A_joint[idx_n, 0:3] = row_north_pole * sw_n

    # 2 Build A
    A_joint[idx_e, 3] = 1.0 * sw_e
    A_joint[idx_e, 4] = 0.0
    A_joint[idx_n, 3] = 0.0
    A_joint[idx_n, 4] = 1.0 * sw_n

    # 3 Build B
    B[idx_e] = (pvData.v_es[i]) * sw_e
    B[idx_n] = (pvData.v_ns[i]) * sw_n

  north_hemisphere = (sum_lats > 0.0)

  Sol_joint, residuals, rank, s = np.linalg.lstsq(A_joint, B, rcond=None)

  Omega_c = Sol_joint[0:3]
  Offset = Sol_joint[3:5]  # mm/yr (km/Ma)

  if (Omega_c[2] > 0) != north_hemisphere:
    Omega_c = -Omega_c

  Omega_mag = np.linalg.norm(Omega_c)
  omega_deg_myr = np.degrees(Omega_mag)

  lat_pole = np.degrees(np.arcsin(Omega_c[2] / Omega_mag))
  lon_pole = np.degrees(np.arctan2(Omega_c[1], Omega_c[0]))

  return EulerPole(lon_pole, lat_pole, omega_deg_myr, is_clockwise=True), Offset


# Early Gemini attempt at a unified regression (rot, offset). 
def fit_euler_pole_linear3(pvData):
  """
  Finds the best-fitting Euler pole and global 3D translational offset
  simultaneously, preventing mathematical cross-talk across wide regions.
  """

  num_stations = len(pvData.lats)

  # 6 Columns: 3 for Rotation [Omega_x, Y, Z], 3 for 3D Translation [T_x, Y, Z]
  A_joint = np.zeros((2 * num_stations, 6))
  B = np.zeros(2 * num_stations)

  sum_lats = 0
  for i in range(num_stations):
    phi = np.radians(pvData.lats[i])
    lam = np.radians(pvData.longs[i])
    sum_lats += pvData.lats[i]

    # weights for this station
    sw_e = 1.0 / pvData.s_es[i] if pvData.s_es is not None else 1.0
    sw_n = 1.0 / pvData.s_ns[i] if pvData.s_ns is not None else 1.0

    # Construct local basis frames
    e_hat = np.array([-np.sin(lam), 
                      np.cos(lam), 
                      0.0])
    
    n_hat = np.array([-np.sin(phi) * np.cos(lam), -
                      np.sin(phi) * np.sin(lam),
                      np.cos(phi)])

    # Build 3D Geocentric station position vector (meters)
    P = R * np.array([np.cos(phi) * np.cos(lam),
                         np.cos(phi) * np.sin(lam), 
                         np.sin(phi)])

    # 1. Rigorous rigid block rotation row entries (O x P)
    row_east_pole = np.cross(P, e_hat)
    row_north_pole = np.cross(P, n_hat)

    idx_e = 2 * i
    idx_n = 2 * i + 1

    # Populate rotation matrix rows (0, 1, 2)
    A_joint[idx_e, 0:3] = row_east_pole * sw_e
    A_joint[idx_n, 0:3] = row_north_pole * sw_n

    # 2. Rigorous 3D Translation row entries (Projecting T onto local station frames)
    A_joint[idx_e, 3:6] = e_hat * sw_e
    A_joint[idx_n, 3:6] = n_hat * sw_n

    # 3. Populate observation target array
    B[idx_e] = pvData.v_es[i] * sw_e
    B[idx_n] = pvData.v_ns[i] * sw_n

  north_hemisphere = (sum_lats > 0.0)

  # Run the joint linear regression pass
  Sol_joint, residuals, rank, s = np.linalg.lstsq(A_joint, B, rcond=None)

  # Isolate parameters cleanly
  Omega_c = Sol_joint[0:3]
  T_cartesian = Sol_joint[3:6]  # Uniform 3D ECEF translation vector (m/Ma)

  if (Omega_c[2] > 0) != north_hemisphere:
    Omega_c = -Omega_c

  Omega_mag = np.linalg.norm(Omega_c)
  omega_deg_myr = np.degrees(Omega_mag)  # Outputs cleanly as °/Myr

  lat_pole = np.degrees(np.arcsin(Omega_c[2] / Omega_mag))
  lon_pole = np.degrees(np.arctan2(Omega_c[1], Omega_c[0]))

  # Return the translation vector alongside the Euler Pole definition
  offset_e = np.dot(e_hat, T_cartesian)
  offset_n = np.dot(n_hat, T_cartesian)
  Offset = np.array([offset_e, offset_n])

  return EulerPole(lon_pole, lat_pole, omega_deg_myr, is_clockwise=True) #, Offset 


def extractEulerPoleUsingCombinedRegressions(pvDataIn, use_stereo=True):

  # get data Euler pole from the raw data set
  raw_pole = fit_euler_pole_linear(pvDataIn)

  # apply Gauss-Newton analysis to get any translation (non-rotation) components
  offset = gn.solve_gauss_newton_transform(pvDataIn, raw_pole, use_stereo)
  # print(f"offset: {offset}")

  for iteration in range(3):
    # strip any translation element to get rot-only
    off_pvData = replace(pvDataIn, v_es = pvDataIn.v_es - offset[0], v_ns = pvDataIn.v_ns - offset[1])
    # rot_ve_list = np.array(pvData.ve) - offset[0]
    # rot_vn_list = np.array(pvData.vn) - offset[1]

    # get data Euler pole from the raw data set
    rot_pole = fit_euler_pole_linear(off_pvData)
    # rot_pole.print("2: rotPole: ")

    pass_offset = gn.solve_gauss_newton_transform(off_pvData, rot_pole, use_stereo)
    offset += pass_offset
    # print(f"offset: {offset}")

  # get Velocity pole PAVel info
  pnwVPAVel = getPAvel(offset[0], offset[1])

  # rot_pole.print("rot_pole: ")
  # pnwVPAVel.print("pnwVPAVel: ")
  return rot_pole, pnwVPAVel


def getPnwGpsRotPoleAndVelocity(center, max_distance, useStereo=True):  # radius km
  pvData = tu.get_GPS_rotation_data(center, max_distance)
  rot_pole, pnwVPAVel = extractEulerPoleUsingCombinedRegressions(pvData, useStereo)
  return rot_pole, pnwVPAVel


def getPnwGpsRotPoleAndVelocity2(pvData, useStereo=True):  
  rot_pole, pnwVPAVel = extractEulerPoleUsingCombinedRegressions(pvData, useStereo)
  return rot_pole, pnwVPAVel


def print_result(name, pole_result, point_count=0):
  print(f"{name} count: {point_count}")
  print(f"Longitude: {pole_result.long:.5f}° E")
  print(f"Latitude:  {pole_result.lat:.5f}° N")
  print(f"Rate:      {pole_result.omega:.5f}° / Myr")
