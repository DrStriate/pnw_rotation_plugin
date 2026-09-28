import os
import numpy as np
import euler_kinematics as ek
import geopandas as gpd
import geo_helper as gh
from geo_helper import R, PAvel, PLoc, EulerPole, pvData

OC_NA_Pole = EulerPole(lat=45.54, long=-119.60, omega=1.32, is_clockwise=True)

# parameters for PWN rot sample data
sample_radius = 600.0  # km
sample_center = PLoc(-119.0, 45.0)


def get_data_file_path(name):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  plugin_root = os.path.dirname(current_dir)
  data_folder_path = os.path.join(plugin_root, "data", name)
  data_folder_path = os.path.normpath(data_folder_path).replace("\\", "/")
  return data_folder_path


def get_test_data():
  return get_GPS_rotation_data(OC_NA_Pole, 600.0)  # km


def get_GPS_rotation_data(center, max_distance):  # distance in km
  gh.setGeod(realWorld=True)

  file_path = get_data_file_path("NSHM2023_GPS_velocity.zip")
  gdf = gpd.read_file(f"/vsizip/{file_path}")
  list_lats = gdf['geometry'].y.values
  list_lons = gdf['geometry'].x.values
  list_v_east = gdf['Ve'].values
  list_v_north = gdf['Vn'].values
  list_s_east = gdf['Se'].values
  list_s_north = gdf['Sn'].values

  lats = []
  lons = []
  v_east = []  # mm/ yr
  v_north = []  # mm/ yr
  s_east = []
  s_north = []

  for i in range(len(list_lats)):
    dist = gh.getDistanceBetweenPoints(
        gh.PLoc(list_lons[i], list_lats[i]), center)
    if dist < max_distance:
      lats.append(list_lats[i])
      lons.append(list_lons[i])
      v_east.append(list_v_east[i])
      v_north.append(list_v_north[i])
      s_east.append(list_s_east[i])
      s_north.append(list_s_north[i])

  return pvData(lons, lats, v_east, v_north, s_east, s_north)


def create_random_sample_ring(euler_pole,
                              sample_ploc,
                              count,
                              max_dist,  # km
                              test_omega = None,
                              crop = 1.0,
                              rms = 0.0):
  rng = np.random.default_rng(seed=42)
  rands = rng.random(size=(count, 2))
  v_noise = rng.normal(loc=0.0, scale=rms, size=(count, 2))

  sample_n = []
  sample_e = []
  sample_v_east = []
  sample_v_north = []  # mm/ yr

  max_long = gh.create_sample(
      sample_ploc.long, sample_ploc.lat, 90.0, max_dist).long
  min_long = gh.create_sample(
      sample_ploc.long, sample_ploc.lat, 270.0, max_dist).long
  long_range = max_long - min_long
  crop_long = min_long + (max_long - min_long) * crop
  cropped_samples = 0

  for i in range(len(rands)):
    sample = gh.create_sample(
        sample_ploc.long, sample_ploc.lat, 360.0 * rands[i][0], max_dist * rands[i][1])
    v = ek.calculate_v_from_EulerPole(
        euler_pole, sample, test_omega) + v_noise[i]

    if sample.long < crop_long:
      sample_e.append(sample.long)
      sample_n.append(sample.lat)
      sample_v_east.append(v[0])
      sample_v_north.append(v[1])

    # print(f"{i}: sample.long: {sample.long:.3f}, sample['lon']: {sample['lon']:.3f}, v_e: {v['v_e']:.2f}  v_n: {v['v_n']:.2f}")
    cropped_samples += 1

  # print(f"samples = {cropped_samples} out of {count}")

  return pvData(sample_e, sample_n, sample_v_east, sample_v_north)


def create_random_sample_dual_pole_ring(euler_pole1,
                                        euler_pole2,
                                        ring_center,
                                        count,
                                        max_dist,  # meters
                                        crop=1.0,
                                        rms=0.0):
  rng = np.random.default_rng(seed=42)
  rands = rng.random(size=(count, 2))
  v_noise1 = rng.normal(loc=0.0, scale=rms, size=(count, 2))
  v_noise2 = rng.normal(loc=0.0, scale=rms, size=(count, 2))

  sample_n = []
  sample_e = []
  sample_v_east = []
  sample_v_north = []  # mm/ yr

  max_long = gh.create_sample(
      ring_center.long, ring_center.lat, 90.0, max_dist).long
  min_long = gh.create_sample(
      ring_center.long, ring_center.lat, 270.0, max_dist).long
  crop_long = min_long + (max_long - min_long) * crop
  cropped_samples = 0

  # must 'normalize' poles w. clockwise rotation intent to use a flipped pole
  for i in range(len(rands)):
    sample = gh.create_sample(
        ring_center.long, ring_center.lat, 360.0 * rands[i][0], max_dist * rands[i][1])
    v1 = ek.calculate_v_from_EulerPole(
        euler_pole1.normalize(), sample) + v_noise1[i]
    v2 = ek.calculate_v_from_EulerPole(
        euler_pole2.normalize(), sample) + v_noise2[i]

    if sample.long < crop_long:
      sample_e.append(sample.long)
      sample_n.append(sample.lat)
      sample_v_east.append(v1[0] + v2[0])
      sample_v_north.append(v1[1] + v2[1])

    # print(f"{i}: sample.long: {sample.long:.3f}, sample['lon']: {sample['lon']:.3f}, v_e: {v['v_e']:.2f}  v_n: {v['v_n']:.2f}")
    cropped_samples += 1

  # print(f"samples = {cropped_samples} out of {count}")

  return pvData(sample_e, sample_n, sample_v_east, sample_v_north)


def distance_for_target_velocity(omega_deg_per_ma, target_v_mm_per_yr=1.0):
    # Convert angular velocity omega to rad/Ma
  omega_rad_per_ma = np.radians(omega_deg_per_ma)

  # Convert target velocity from mm/Ma to m/Ma
  v_m_per_ma = target_v_mm_per_yr * 1000.0

  # Linear velocity on Earth's surface: v = omega * R * sin(delta)
  # So sin(delta) = v / (omega * R)
  sin_delta = v_m_per_ma / (omega_rad_per_ma * R * 1000)

  if sin_delta > 1.0:
    raise ValueError(
        "Target velocity is too large for the given rotation rate.")

  delta_rad = np.arcsin(sin_delta)
  d_m = R * delta_rad
  return d_m

# dist in km


def create_simple_sample_quad(euler_pole, azimuths, dist, realWorld=False):
  return create_simple_sample_quad_w_trans(euler_pole, [0, 0], azimuths, dist, realWorld)


def create_simple_sample_quad_w_trans(euler_pole, v_trans, azimuths, dist, realWorld=False):
  longs = np.zeros(4)
  lats = np.zeros(4)
  v_easts = np.zeros(4)
  v_norths = np.zeros(4)

  Omega = {"omega": euler_pole.omega, "phi": np.radians(
      euler_pole.lat), "lamb": np.radians(euler_pole.long)}

  # Calculate angular radius in radians (distance / Earth radius)
  angular_radius = dist / R

  for i in range(len(azimuths)):
    az_rad = np.radians(azimuths[i])

    # Spherically calculate points strictly relative to the pole coordinates
    # This ensures perfect small-circle symmetry
    sample_lat_rad = np.arcsin(np.sin(Omega['phi']) * np.cos(angular_radius) +
                               np.cos(Omega['phi']) * np.sin(angular_radius) * np.cos(az_rad))

    sample_lon_rad = Omega['lamb'] + np.arctan2(np.sin(az_rad) * np.sin(angular_radius) * np.cos(Omega['phi']),
                                                np.cos(angular_radius) - np.sin(Omega['phi']) * np.sin(sample_lat_rad))

    sample_lat = np.degrees(sample_lat_rad)
    sample_lon = np.degrees(sample_lon_rad)

    p = {"phi": sample_lat_rad, "lamb": sample_lon_rad}
    v = calculate_v_from_Euler_pole(Omega, p, Omega['omega'], realWorld)

    # print(f"{i}: sample.long: {sample_lon:.3f}, sample.lat: {sample_lat:.3f}, v_e: {v['v_e']:.2f}  v_n: {v['v_n']:.2f}")

    longs[i] = sample_lon
    lats[i] = sample_lat
    v_easts[i] = v['v_e'] + v_trans[0]
    v_norths[i] = v['v_n'] + v_trans[1]

  return pvData(longs, lats, v_easts, v_norths)


def calculate_v_from_Euler_pole(Omega, p, omega, realWorld):
  if realWorld:
      # p: dict with {'phi': lat_rad, 'lamb': lon_rad}
      # Omega: dict with {'phi': pole_lat_rad, 'lamb': pole_lon_rad}
      # omega: scalar rotation rate (degrees/Myr or rad/yr depending on your scaling)

      # 1. Compute the local radius of curvature along the prime vertical (R_N)
      # This accounts for the ellipsoidal bulge at the station's exact latitude
    R_N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(p['phi'])**2)

    # 2. Build the 3D position vector using the local radius
    # For horizontal velocities, assume ellipsoidal height h = 0
    P = R_N * get_hat_p(p)

    # 3. Convert angular velocity to radians per unit time and build rotation vector
    omega_rad = np.radians(omega)
    O = omega_rad * get_hat_p(Omega)

    # 4. Standard rigid cross product
    # Note: O x P yields the standard right-hand velocity vector
    V = np.cross(O, P)

    # 5. Project 3D vector to local East and North ellipsoidal vectors
    v = project_V_to_v(V, p)
    return v
  else:  # Idealized: usually functional test
    P = R * get_hat_p(p)
    O = np.radians(omega) * get_hat_p(Omega)
    V = np.cross(O, P)  # Standard kinematic rotation vector (Omega x P)
    v = project_V_to_v(V, p)
    return v

# Calculates omega rotation around euler_pole as experienced by p_loc as {v_e, v_n} dict velocity


# omaga is angle of pole rotation
def calculate_v_from_Euler_pole2(euler_pole, ploc, omega, realWorld):
  Omega = {"omega": euler_pole.omega, "phi": np.radians(
      euler_pole.lat), "lamb": np.radians(euler_pole.long)}
  p = {"phi": np.radians(ploc.lat), "lamb": np.radians(ploc.long)}
  v = calculate_v_from_Euler_pole(Omega, p, omega, realWorld)
  return v


def project_V_to_v(V, p):  # V is 3D cartesion velocity, phi and lamb in radians
  e_hat = np.array([-np.sin(p['lamb']), np.cos(p['lamb']), 0])
  n_hat = np.array([-np.sin(p['phi']) * np.cos(p['lamb']), -
                   np.sin(p['phi']) * np.sin(p['lamb']), np.cos(p['phi'])])
  v_e = np.dot(V, e_hat)
  v_n = np.dot(V, n_hat)
  return {"v_e": v_e, "v_n": v_n}


def get_hat(lat, long):
  return get_hat_p({'lamb': np.radians(long), 'phi': np.radians(lat)})


# WGS84 Constants required for geodetic-to-cartesian conversions
WGS84_A = 6378137.0         # Semi-major axis (meters)
WGS84_E2 = 0.00669437999014  # First eccentricity squared


def get_hat_p(p):
  # Returns a unit normal vector to the geodetic phi, lamb point
  return np.array([
      np.cos(p['phi']) * np.cos(p['lamb']),
      np.cos(p['phi']) * np.sin(p['lamb']),
      np.sin(p['phi'])
  ])

# Testing regression inversions like  x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)


def test_regression_stats(x, A, B, residuals, verbose=False):
  if residuals.size == 0:
    y_pred = A @ x
    ssr = np.sum((B - y_pred) ** 2)
    print("")
    print(f"Sum of Squared Residuals (SSR): {ssr:.4f}")
  else:
    ssr = residuals[0]  # Sum of squared residuals

    # Total sum of squares
    tss = np.sum((B - np.mean(B)) ** 2)

    # R-squared (R2)
    r_squared = 1 - (ssr / tss)

    # Root Mean Squared Error (RMSE)
    n = len(B)
    rmse = np.sqrt(ssr / n)

    if verbose or test_verbose:
      print("")
      print(f"Sum of Squared Residuals (SSR): {ssr:.4f}")
      print(f"R-squared (Goodness of Fit): {r_squared:.4f}")
      print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


test_verbose = False
