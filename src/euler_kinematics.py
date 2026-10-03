import numpy as np
import geo_helper as gh
from geo_helper import PLoc, EulerPole, geod, R

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
def getPoleRotationOfPointSphere(pole, ploc, ma):

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


def getPoleRotationOfPoint(pole, pLoc, ma, realWorld):
    """
    Rotates a point around an Euler pole on an ellipsoidal Earth (e.g. WGS84).
    
    Parameters:
      pole: Object with attributes 'lat', 'lon' (or ploc() returning [lat, lon]), and 'omega' (deg/Ma).
      ploc: Tuple or list of [latitude, longitude] in degrees.
      ma: Time duration in Million Years (or time step units matching omega).
      geod: pyproj.Geod instance (defaults to WGS84).
      
    Returns:
      ploc2: [latitude, longitude] of the rotated point in degrees.
    """

    print(f"realWorld: {realWorld}")

    if not realWorld:
      return getPoleRotationOfPointSphere(pole, pLoc, ma)

    #if pole.is_clockwise:
    ma = -ma

    gh.setGeod(realWorld)
    
    # 1. Inverse geodesic calculation: find azimuth and distance from pole to point
    # inv returns: (azimuth_forward, azimuth_back, distance_meters)
    az_pole_to_pt, _, dist_meters = geod.inv(pole.long, pole.lat, pLoc.long, pLoc.lat)
    
    # 2. Calculate the angular rotation around the pole (in degrees)
    delta_theta = pole.omega * ma
    
    # Update the azimuth around the Euler pole
    new_azimuth = (az_pole_to_pt + delta_theta) % 360.0
    
    # 3. Forward geodesic calculation: find new point location at same distance
    # fwd returns: (new_lon, new_lat, back_azimuth)
    new_lon, new_lat, _ = geod.fwd(pole.long, pole.lat, new_azimuth, dist_meters)
    
    return PLoc(new_lon, new_lat)

# Combo pole emulation (R-V ordering)
def getCompoundRotationTranslationOfPoint(vPole, rPole, ploc, ma, realWorld):
  # move the rot pole to the proper loc for ma
  rot_pole_ma_ploc = getPoleRotationOfPoint(vPole, rPole.ploc(), ma, realWorld)
  ma_rot_pole = gh.EulerPole(
      rot_pole_ma_ploc.long, rot_pole_ma_ploc.lat, rPole.omega, is_clockwise=True)

  # Rotate by ma scaled rot pole omega
  loc_2 = getPoleRotationOfPoint(ma_rot_pole, ploc, ma, realWorld)

  # Move rotated point up according to vPole ma
  loc_3 = getPoleRotationOfPoint(vPole, loc_2, -ma, realWorld)
  return loc_3

# This is a very old (and naive) interpretation of NA plate motion using NA PAVel, 
# We should revise to use the new model - for now it's only used in test_3_pole_50ma_yhs_movement
# Track yhs from ploc using NA plate motion and PNW rotation and velocity info
def getPlocFromPoleData(naPAvel, pnwRotPole, pnwVPavel, ploc, ma):
  realWorld=False
  pnwVPole = getEulerPoleFromPlocAndPavel(pnwRotPole.ploc(), pnwVPavel)
  naPole = getEulerPoleFromPlocAndPavel(ploc, naPAvel)

  # move NA over yhs then move by both pnw poles to its ma location
  loc_2 = getPoleRotationOfPoint(naPole, ploc, -ma, realWorld)
  loc_3 = getCompoundRotationTranslationOfPoint(pnwVPole, pnwRotPole, loc_2, ma, realWorld)
  return loc_3

# cartesian 3D Velocity vector for ploc and its pAvel motion azimuth and magnitude (mm/yr)
def getVFromAzvel(pLoc, pAvel):
  # 2D motion at point on pLoc tangent plane
  v_e = np.sin(np.radians(pAvel.azimuth)) * pAvel.vel
  v_n = np.cos(np.radians(pAvel.azimuth)) * pAvel.vel
  # return net scaled velocity in easterly and northerly directions on pLoc tangent plane
  return e_hat(pLoc) * v_e + n_hat(pLoc) * v_n

# Big circle pole for given loc and velocity vector. pAvel is azimuth and speed (e.f. km/ma or mm/yr)
# Used for defining the v-pole of the coupled pole model 
def getEulerPoleFromPlocAndPavel(ploc, pAvel):
  KmPerDegree = R * np.pi /  180.0
  omega = pAvel.vel / KmPerDegree # deg/ma, pAvel in km/Ma

  P = getVectFromPloc(ploc)
  V = getVFromAzvel(ploc, pAvel) # 3D V vector defined by ploc tangent plane and pAvel velocity vector on that plane
  
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
