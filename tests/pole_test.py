from geo_helper import EulerPole, R, PLoc, pvData  # R is earth radius
import geo_helper as gh
import euler_pole_regression as epr
import gauss_newton as gn
import test_utils as tu
import pytest
import numpy as np


def test_quad_pole():
  gh.setGeod(realWorld=False)
  # long, lat, omega
  euler_pole = EulerPole(-99, 45.0, 1.32, is_clockwise=True)
  azimuths = [45.0, 135.0, 225.0, 315.0]  # directions  to test lat/long
  sample_dist = 50000  # m
  realWorld = False
  # print("")

  #  Create samples, regress to pole
  testPvData = tu.create_simple_sample_quad(euler_pole, azimuths, sample_dist, realWorld)
  pole_result = epr.fit_euler_pole_linear(testPvData)
  # pole_result.print("pole_result")

  v_offset = gn.solve_gauss_newton_transform(testPvData, pole_result, useStereo=False)
  # print(f"v_offset1: {v_offset}")

  assert pole_result.omega == pytest.approx(euler_pole.omega, abs=1e-8)
  assert pole_result.lat == pytest.approx(euler_pole.lat, abs=1e-8)
  assert pole_result.long == pytest.approx(euler_pole.long, abs=1e-8)
  assert v_offset[0] == pytest.approx(0.0, abs=1e-8)
  assert v_offset[1] == pytest.approx(0.0, abs=1e-8)

# demonstrating new pole-offset results in same pole when v's moved by constant and good extraction


def test_offset_quad_pole():
  gh.setGeod(realWorld=False)
  # long, lat, omega
  euler_pole = EulerPole(-99, 45.0, 1.32, is_clockwise=True)
  azimuths = [45.0, 135.0, 225.0, 315.0]  # directions  to test lat/long
  sample_dist = 50  # km
  realWorld = False

  #  Create samples, regress to pole
  testPvData = tu.create_simple_sample_quad(euler_pole, azimuths, sample_dist, realWorld)

  # relatively inaccurate global v change: using a pole would be more accurate
  sample_vs = [0.001, 0.002]
  testPvData.v_es += sample_vs[0]
  testPvData.v_ns += sample_vs[1]

  pole_result = epr.fit_euler_pole_linear(testPvData)
  # pole_result.print("pole_result")

  v_offset = gn.solve_gauss_newton_translation(testPvData, pole_result)
  # print(f"v_offset1: {v_offset}")

  assert pole_result.omega == pytest.approx(euler_pole.omega, abs=1e-5)
  assert pole_result.lat == pytest.approx(euler_pole.lat, abs=1e-3)
  assert pole_result.long == pytest.approx(euler_pole.long, abs=0.002)
  assert v_offset[0] == pytest.approx(sample_vs[0], rel=0.002)
  assert v_offset[1] == pytest.approx(sample_vs[1], rel=1e-3)

# repo of 'decomposed' regression we used to iterate to get offset. Note sb pole


def test_euler_GPS_pole_extraction_legacy():
  gh.setGeod(realWorld=True)
  center_loc = PLoc(-118.0, 45.0)
  max_distance = 600  # km
  gpsPvData = tu.get_GPS_rotation_data(center_loc, max_distance)

  pole_result = epr.fit_euler_pole_linear(gpsPvData)
  # epr.print_result ("test_GPS_pole_extraction", pole_result, len(lats))

  pole_result_sb = EulerPole(-115.406, 43.646,  0.550)
  assert pole_result.long == pytest.approx(pole_result_sb.long, abs=0.001)
  assert pole_result.lat == pytest.approx(pole_result_sb.lat, abs=0.001)
  assert pole_result.omega == pytest.approx(pole_result_sb.omega, abs=0.001)


def test_euler_GPS_pole_extraction2():
  gh.setGeod(realWorld=True)
  center_loc = PLoc(-118.0, 45.0)
  max_distance = 600  # km

  # 1. Elements loaded cleanly in unscaled mm/yr
  gpsPvData = tu.get_GPS_rotation_data(center_loc, max_distance)

  # 2. Use your pristine baseline legacy pole calculator (which maps rad/yr perfectly if R is in mm)
  # Legacy expected mm/yr and R in meters? -> R_km scale matches your legacy wz extraction.
  pole_result = epr.fit_euler_pole_linear(gpsPvData)
  # pole_result.print("Pristine Euler Pole")

  # 3. Pass parameters into the robust translation solver to extract the net sheet motion
  v_offset_mm_yr = gn.solve_gauss_newton_translation(gpsPvData, pole_result)
  # print(f"Isolated Average Background Velocity (mm/yr East, North): {v_offset_mm_yr}")

  sb = np.array([0.817,  2.935])
  assert v_offset_mm_yr == pytest.approx(sb, abs=0.001)
