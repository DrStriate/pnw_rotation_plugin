import numpy as np
import geo_helper as gh
from geo_helper import PLoc, EulerPole, R

# get cartesian vector for PLoc
# returns [x, y, z] (unit vector normal to p)
def getVectFromPloc(p):
  return np.array([
      np.cos(p.phi) * np.cos(p.lam),
      np.cos(p.phi) * np.sin(p.lam),
      np.sin(p.phi)
  ])

def getPlocFromVector(r):
  lamb = np.arctan2(r[1], r[0])
  phi = np.arcsin(r[2])
  return PLoc(np.degrees(lamb), np.degrees(phi))

# get omega vector from pole (normalized - unscaled by pole omega)
def getWVector(pole):
  return getVectFromPloc(pole.ploc())

 
# unit vectors for 'easterly' and 'northerly' at P
def e_hat(pLoc):
  return np.array([-np.sin(pLoc.lam), np.cos(pLoc.lam), 0.0])
def n_hat(pLoc):
  return np.array([-np.sin(pLoc.phi) * np.cos(pLoc.lam), -
                   np.sin(pLoc.phi) * np.sin(pLoc.lam), np.cos(pLoc.phi)])

def getPlocFromLocNormal(p_hat):
  phi = np.arcsin(p_hat[2])
  lam = np.arctan2(p_hat[1], p_hat[0])
  return PLoc(np.degrees(lam), np.degrees(phi))

# Rotates point ploc around Euler pole by omega * ma to a new ploc2 
def getPoleRotationOfPoint(pole, ploc, ma):

  # Apply Rodrigues' rotation formula
  theta = np.radians(pole.omega) * ma
  R = getVectFromPloc(ploc)  
  W = getVectFromPloc(pole.ploc()) 
  W_cross_R = np.cross(W, R)
  W_dot_R = np.dot(W, R)
  V_new = R * np.cos(theta) + W_cross_R * np.sin(theta) + \
      W * W_dot_R * (1 - np.cos(theta))

  # Convert the rotated Cartesian vector back to lat/long
  return getPlocFromVector(V_new)

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

# Big circle pole for given loc and velocity vector. pAvel is azimuth and speed (e.f. km/ma or mm/yr)
# cartesian Ve and Vn for point, and motion azimuth and magnitude (mm/Y)
def getVeVnFromAzvel(pLoc, pAzvel):
  # 2D motion vector at point
  V = np.array([np.sin(np.radians(pAzvel.azimuth)) * pAzvel.vel,
                np.cos(np.radians(pAzvel.azimuth)) * pAzvel.vel])
  # return scaled velocity in easterly and northerly directions
  return e_hat(pLoc) * V[0], n_hat(pLoc) * V[1]

# Get pole defined by a great-circle motion at plic and with velocity and direction of pAvel
# Used for defining the v-pole of the coupled pole model
def getEulerPoleFromPlocAndPavel(ploc, pAvel):
  MetersPerDegree = 2 * np.pi * R / 360.0
  KmPerMaToDegreesPerMa = 1.0 / MetersPerDegree
  degreesPerMa = pAvel.vel * KmPerMaToDegreesPerMa
  omega = degreesPerMa

  P = getVectFromPloc(ploc)
  V_e, V_n = getVeVnFromAzvel(ploc, pAvel)
  V = V_e + V_n
  pe_hat = gh.normalize(np.cross(P, V))  # epipolar unit direction vector
  epiPoleLoc = getPlocFromLocNormal(pe_hat)
  return EulerPole(epiPoleLoc.long, epiPoleLoc.lat, omega)

def getVForPlocFromPole(pole, pLoc, omega=None):
  P = R * getVectFromPloc(pLoc)
  O = np.radians(pole.omega if omega is None else omega) * \
      getVectFromPloc(pole.ploc())
  V = np.cross(O, P)
  v_e = np.dot(V, e_hat(pLoc))
  v_n = np.dot(V, n_hat(pLoc))
  return np.array([v_e, v_n])
