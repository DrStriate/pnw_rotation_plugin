import geo_helper as gh
import pytest
import test_utils as tu

NA_Speed = 35e-3    # m / yr (Current)
NA_azimuth = 240.0  # degrees azimuth
YHS_lat = 44.43     # Yellowstone hotspot caldera
YHS_long = -110.67
JdF_Lat = 48.25     # Center of Strait of Juan de Fuca
JdF_Long = -124.0


def test_latitudeFromDistance():
  distN = 1.0  # km
  latSb = 0.00898  # degrees
  lat = gh.latitudeFromDistN(distN)
  assert lat == pytest.approx(latSb, abs=0.001)


def test_lonitudeFromDistance():
  distN = 1.0  # km
  lat = 45
  lonSb = 0.0127  # degrees
  lon = gh.longitudeFromDistE(lat, distN)
  assert lon == pytest.approx(lonSb, abs=0.001)
