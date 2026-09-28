import numpy as np
from pyproj import Geod, CRS, Transformer
from dataclasses import dataclass, field

R = 6371.0  # Earth radius in km

geod = Geod(ellps="WGS84")
isRealWorld = True

def setGeod(realWorld):
  global geod, isRealWorld
  isRealWorld = realWorld
  if realWorld:
    geod = Geod(ellps="WGS84")
  else:
    geod = Geod(a=R*1e3, b=R*1e3)


def kmPerDegree():
  return R * np.pi / 180.0


def getMagnitude(d_e, d_n):    # only use for very small distances
  return np.sqrt(d_e * d_e + d_n * d_n)


def getAzimuth(v_e, v_n):
  azimuth = np.degrees(np.arctan2(v_e, v_n))
  if azimuth < 0.0:
    azimuth += 360.0  # arctan2 returns -180 ... 180 wheras azimuth needs to be 0 ... 360
  return azimuth


def getPAvel(v_e, v_n):
  return PAvel(getAzimuth(v_e, v_n), getMagnitude(v_e, v_n))


@dataclass
class PLoc:
  long: float
  lat: float
  phi: float = field(init=False)
  lam: float = field(init=False)

  def __post_init__(self):
    self.phi = np.radians(self.lat)
    self.lam = np.radians(self.long)

  def print(self, label=""):
    print(f"{label} long: {self.long:0.3f}, lat:  {self.lat:0.3f}")

  def __iadd__(self, other):  # Overriding the += operator
    if isinstance(other, PLoc):
      self.long += other.long
      self.lat += other.lat
      self.lam += other.lam
      self.phi += other.phi
      return self
    return NotImplemented

@dataclass
# PVel should only be used for (linear) Velocity. For V the standard is mm/Yr (= km/ma)
# distance measurements in plate kinematics should be in degrees, not kilometers.
class PVel:
  east: float
  north: float

  def azimuth(self):
    return getAzimuth(self.east, self.north)

  def magnitude(self):  # only valid for very small magnitudes
    return getMagnitude(self.east, self.north)

  def print(self, label=""):
    print(f"{label} east: {self.east:0.3f}, north:  {self.north:0.3f}")

@dataclass
# PAvel azimuth is in degrees (cw from N) and vel is in  mm/yr or km/Ma (equiv)
# Note that distance measurements in plate kinematics should be in degrees, not kilometers.
class PAvel:
  azimuth: float
  vel: float

  def print(self, label=""):
    print(f"{label} azimuth: {self.azimuth:0.3f}, vel:  {self.vel:0.3f}")

  @classmethod
  def from_V(cls, v2) -> 'PAvel':  # V2 is [east_v, north_v]
    return cls(
        np.degrees(np.arctan2(v2[0], v2[1])),
        np.hypot(v2[0], v2[1]))

@dataclass
class EulerPole:
  long: float
  lat: float
  omega: float

  is_clockwise: bool = False  # Set to True for clockwise poles needing antipodal shift

  def ploc(self):
    return PLoc(self.long, self.lat)

  def print(self, label: str = ""):
    print(f"{label} long: {self.long:0.3f}, lat: {self.lat:0.3f}, omega: {self.omega:.6f}")

  def normalize(self) -> 'EulerPole':
    """
    Ensures right-hand rule math produces the correct surface motion.
    If rotation is clockwise, returns a NEW EulerPole flipped to its 
    antipodal position with is_clockwise=False so standard CCW vector math works.
    """
    if self.is_clockwise:
      # Shift longitude by 180 and keep within [-180, 180]
      new_long = (self.long + 180) % 360
      if new_long > 180:
        new_long -= 360

      return EulerPole(
          long=new_long,
          lat=-self.lat,
          omega=self.omega,
          is_clockwise=False
      )
    return self

# raw velocity data - usually either GPS or simulation source
@dataclass
class pvData:
  longs: float
  lats: float
  v_es: float
  v_ns: float
  s_es: float = None
  s_ns: float = None

# Get point from PAvel new point pavel * ma distant
def getPointFromPavel(start_point, pAVel, ma):
  """
  Calculates the destination latitude/longitude using a spherical Earth model.
  """
  # Convert degrees to radians
  lat1 = np.radians(start_point.lat)
  lon1 = np.radians(start_point.long)
  azimuth = np.radians(pAVel.azimuth)

  # Angular distance covered
  angular_vel = (pAVel.vel * ma) / R

  # Calculate destination latitude
  lat2 = np.arcsin(np.sin(lat1) * np.cos(angular_vel) +
                   np.cos(lat1) * np.sin(angular_vel) * np.cos(azimuth))

  # Calculate destination longitude
  lon2 = lon1 + np.arctan2(np.sin(azimuth) * np.sin(angular_vel) * np.cos(lat1),
                           np.cos(angular_vel) - np.sin(lat1) * np.sin(lat2))

  # Convert back from radians to degrees
  destination_lat = np.degrees(lat2)
  destination_lon = np.degrees(lon2)

  # Normalize longitude to be between -180 and +180
  destination_lon = (destination_lon + 540) % 360 - 180

  return PLoc(destination_lon, destination_lat)

# returns km


def getNortherlyEasterlyFromLatLongPoints(lon1, lat1, lon2, lat2):
  # forward_azimuth is the angle from point 1 to point 2 (degrees clockwise from North)
  forward_azimuth, back_azimuth, distance_meters = geod.inv(
      lon1, lat1, lon2, lat2)

  # Convert azimuth to radians
  azimuth_rad = np.radians(forward_azimuth)

  # Calculate components
  northerly_km = distance_meters * np.cos(azimuth_rad) * 0.001
  easterly_km = distance_meters * np.sin(azimuth_rad) * 0.001
  return northerly_km, easterly_km


def getFwdAzimuthFromLocations(point1, point2):  # both PLocs
   # forward_azimuth is the angle from point 1 to point 2 (degrees clockwise from North)
  forward_azimuth, back_azimuth, distance_meters = geod.inv(
      point1.long, point1.lat, point2.long, point2.lat)
  return forward_azimuth % 360

# Great circle distance (km)


def getDistanceBetweenPoints(point1, point2):  # both PLocs
  forward_azimuth, back_azimuth, distance_meters = geod.inv(
      point1.long, point1.lat, point2.long, point2.lat)
  return distance_meters / 1000.0


def create_sample(start_lon, start_lat, azimuth, distance):  # distance in km!
  # Calculate the terminus point
  end_lon, end_lat, back_azimuth = geod.fwd(
      start_lon,
      start_lat,
      azimuth,
      distance * 1000.0)
  return PLoc(end_lon, end_lat)


def clamp(value, minimum, maximum):
  return max(minimum, min(value, maximum))


def normalize(vect):
  mag = np.linalg.norm(vect)
  if mag > 0:
    return vect / mag
  return vect

# Code below probably needs to be refactored to use code/methods above which are more accurate

# gets lat and long converted to coordinate distances from pole. This is an approximation


def getSamplePoints(long_list, lat_list, center_ploc):
  # convert sample points to km
  p_e, p_n = latlon_to_stereographic(
      lat_list, long_list, center_ploc.lat, center_ploc.long)
  return p_e, p_n


def latlon_to_stereographic(lats, lons, center_lat, center_lon):
  """
  Transforms latitude and longitude arrays to Stereographic (X, Y) grid coordinates in km.

  Parameters:
      lats, lons: array-like or floats of point coordinates in degrees (WGS84).
      center_lat, center_lon: projection origin center point in degrees.

  Returns:
      x, y: grid coordinates in km relative to origin (0, 0).
  """
  if not isRealWorld:
    return stereographic_forward_spherical(lats, lons, center_lat, center_lon)

  # Define custom Oblique Stereographic projection centered on (center_lat, center_lon)
  proj_crs = CRS.from_proj4(
      f"+proj=stere +lat_0={center_lat} +lon_0={center_lon} +k=1.0 "
      f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
  )

  # Create transformer from WGS84 geographic (EPSG:4326) to custom Stereographic
  transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)

  # Transform (always_xy=True expects longitude first, then latitude)
  x, y = transformer.transform(lons, lats)
  return np.array(x) * 1e-3, np.array(y) * 1e-3


def stereographic_forward_spherical(lats, lons, center_lat, center_lon):
  """
  Direct mathematical Oblique Stereographic projection (spherical model).
  """
  # Convert degrees to radians
  phi = np.radians(lats)
  lam = np.radians(lons)
  phi_0 = np.radians(center_lat)
  lam_0 = np.radians(center_lon)

  # Angular distance / scaling factor k
  sin_phi = np.sin(phi)
  cos_phi = np.cos(phi)
  sin_phi0 = np.sin(phi_0)
  cos_phi0 = np.cos(phi_0)
  cos_dlam = np.cos(lam - lam_0)

  # Scale factor k for stereographic tangent plane projection
  k = 2.0 / (1.0 + sin_phi0 * sin_phi + cos_phi0 * cos_phi * cos_dlam)

  x = R * k * cos_phi * np.sin(lam - lam_0)
  y = R * k * (cos_phi0 * sin_phi - sin_phi0 * cos_phi * cos_dlam)

  return x, y


# --- Example Usage ---
if __name__ == "__main__":
  # Center point for the local region
  origin_lat, origin_lon = 45.0, -122.0

  # Points to project
  sample_lats = np.array([45.0, 45.1, 45.0, 44.9])
  sample_lons = np.array([-122.0, -122.0, -121.9, -122.0])

  x, y = latlon_to_stereographic(
      sample_lats, sample_lons, origin_lat, origin_lon)

  print("X Grid Coordinates (meters):", x)
  print("Y Grid Coordinates (meters):", y)

  xs, ys = stereographic_forward_spherical(
      sample_lats, sample_lons, origin_lat, origin_lon)

  print("X Grid Coordinates (meters):", xs)
  print("Y Grid Coordinates (meters):", ys)

# Be wary of use of these distance metrics.


def latitudeFromDistN(dist):  # dist in km North
  lat = np.arctan2(dist, R) * 180.0 / np.pi
  return lat


def longitudeFromDistE(latitude, dist):  # km East
  latitudeRadians = np.radians(latitude)
  radiusOfParallel = R * np.cos(latitudeRadians)  # m
  longitudeDeltaRadians = dist / radiusOfParallel
  return np.degrees(longitudeDeltaRadians)
