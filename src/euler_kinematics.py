import numpy as np
import geo_helper as gh
from geo_helper import PLoc, EulerPole, R

# get cartesian vector for PLoc
# returns [x, y, z] (unit vector normal to p)
def getRVector(p):
  return np.array([
      np.cos(p.phi) * np.cos(p.lam),
      np.cos(p.phi) * np.sin(p.lam),
      np.sin(p.phi)
  ])

def getPFromRVector(r):
  lamb = np.arctan2(r[1], r[0])
  phi = np.arcsin(r[2])
  return PLoc(np.degrees(lamb), np.degrees(phi))

# get omega vector from pole (normalized - unscaled by pole omega)
def getWVector(pole):
  return getRVector(pole.ploc())

# Rotates point ploc around Euler pole by omega * ma to a new ploc2 
def getPoleRotationOfPoint(pole, ploc, ma):

  # Apply Rodrigues' rotation formula
  theta = np.radians(pole.omega) * ma
  v = getRVector(ploc)  # v
  k = getWVector(pole)  # k
  k_cross_v = np.cross(k, v)
  k_dot_v = np.dot(k, v)
  v_new = v * np.cos(theta) + k_cross_v * np.sin(theta) + \
      k * k_dot_v * (1 - np.cos(theta))

  # Convert the rotated Cartesian vector back to lat/long
  return getPFromRVector(v_new)

# Combo pole emulation (R-V ordering)
def getCompoundRotationTranslationOfPoint(vPole, rPole, ploc, ma):

  # move the rot pole to the proper loc for ma
  rot_pole_ma_ploc = getPoleRotationOfPoint(vPole, rPole.ploc(), ma)
  ma_rot_pole = gh.EulerPole(
      rot_pole_ma_ploc.long, rot_pole_ma_ploc.lat, rPole.omega, is_clockwise=True)

  # Rotate by ma scaled rot pole omega
  loc_2 = getPoleRotationOfPoint(ma_rot_pole, ploc, ma)

  # Move rotated point up according to vPole ma
  loc_3 = getPoleRotationOfPoint(vPole, loc_2, -ma)
  return loc_3

# This is a very old (and naive) interpretation of NA plate motion using NA PAVel, 
# We should revise to use the new model - for now it's only used in test_3_pole_50ma_yhs_movement
# Track yhs from ploc using NA plate motion and PNW rotation and velocity info
def getPlocFromPoleData(naPAvel, pnwRotPole, pnwVPavel, ploc, ma):
  pnwVPole = getEulerPoleFromPlocAndPavel(pnwRotPole.ploc(), pnwVPavel)
  naPole = getEulerPoleFromPlocAndPavel(ploc, naPAvel)

  # move NA over yhs then move by both pnw poles to its ma location
  loc_2 = getPoleRotationOfPoint(naPole, ploc, -ma)
  loc_3 = getCompoundRotationTranslationOfPoint(
      pnwVPole, pnwRotPole, loc_2, ma)
  return loc_3

def getPlocFromLocNormal(p_hat):
  phi = np.arcsin(p_hat[2])
  lam = np.arctan2(p_hat[1], p_hat[0])
  return PLoc(np.degrees(lam), np.degrees(phi))

# Big circle pole for given loc and velocity vector. pAvel is azimuth and speed (e.f. km/ma or mm/yr)
# cartesian Ve and Vn for point, and motion azimuth and magnitude (mm/Y)
def getVeVnFromAzvel(pLoc, pAzvel):
  # unit vectors for 'easterly' and 'northerly' at P
  e_hat = np.array([-np.sin(pLoc.lam), np.cos(pLoc.lam), 0.0])
  n_hat = np.array([-np.sin(pLoc.phi) * np.cos(pLoc.lam), -
                   np.sin(pLoc.phi) * np.sin(pLoc.lam), np.cos(pLoc.phi)])
  # 2D motion vector at point
  V = np.array([np.sin(np.radians(pAzvel.azimuth)) * pAzvel.vel,
                np.cos(np.radians(pAzvel.azimuth)) * pAzvel.vel])
  # return scaled velocity in easterly and northerly directions
  return e_hat * V[0], n_hat * V[1]

def getEulerPoleFromPlocAndPavel(ploc, pAvel):
  MetersPerDegree = 2 * np.pi * R / 360.0
  KmPerMaToDegreesPerMa = 1.0 / MetersPerDegree
  degreesPerMa = pAvel.vel * KmPerMaToDegreesPerMa
  omega = degreesPerMa

  P = getRVector(ploc)
  V_e, V_n = getVeVnFromAzvel(ploc, pAvel)
  V = V_e + V_n
  pe_hat = gh.normalize(np.cross(P, V))  # epipolar unit direction vector
  epiPoleLoc = getPlocFromLocNormal(pe_hat)
  return EulerPole(epiPoleLoc.long, epiPoleLoc.lat, omega)

def testIfBigCircleCoplanarity(ploc1, ploc2, ploc3):  # (r1 x r2) dot r3 == 0
  r1 = getRVector(ploc1)
  r2 = getRVector(ploc2)
  r3 = getRVector(ploc3)

  test = np.linalg.cross(r1, r2).dot(r3)
  return test

def project_V_to_v(V, p):  # V is 3D cartesion velocity, p is PLoc
  e_hat = np.array([-np.sin(p.lam), np.cos(p.lam), 0])
  n_hat = np.array([-np.sin(p.phi) * np.cos(p.lam), -
                   np.sin(p.phi) * np.sin(p.lam), np.cos(p.phi)])
  v_e = np.dot(V, e_hat)
  v_n = np.dot(V, n_hat)
  return np.array([v_e, v_n])

# BUG? - not clear why the sign is flipped on V (see test_v_pole_from_sample_point)

# pole is EulerPole, p is ploc, omega in degrees
def calculate_v_from_EulerPole(pole, p, omega=None):
  P = R * getRVector(p)
  O = np.radians(pole.omega if omega is None else omega) * \
      getRVector(pole.ploc())
  V = np.cross(P, O)
  v = project_V_to_v(V, p)
  return -v
