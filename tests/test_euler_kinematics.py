import numpy as np
import pytest
import test_base as tb
import euler_pole_regression as epr
import euler_kinematics as ek
import test_utils as tu
import geo_helper as gh
from geo_helper import PLoc, PAvel, EulerPole, R
import gauss_newton as gn

OC_NA_Pole = EulerPole(-119.60, 45.54, 1.32, is_clockwise=True)
SeattlePloc = PLoc(-122.3321, 47.6062)


def test_rot_pole_from_quad():
  gh.setGeod(realWorld=False)

  euler_pole = OC_NA_Pole  # {"lat" : 45.0,  "long" : -90, "omega" : 1.32 }
  azimuths = [45.0, 135.0, 225.0, 315.0]
  sample_dist = 50  # km

  test_pvData = tu.create_simple_sample_quad(euler_pole, azimuths, sample_dist)
  pole_result = epr.fit_euler_pole_linear(test_pvData)
  # ek.print_result ("test_euler_pole_from_quad", pole_result)

  # BUG - The new geo-correct model that (now) works with GPS is pretty bad with quad test. See second version for legacy
  #gn_out = gn.solve_gauss_newton_2D_transform(test_pvData, pole_result.ploc())
  #gn_out2 = gn.solve_gauss_newton_translation(test_pvData, pole_result)

  assert pole_result.omega == pytest.approx(euler_pole.omega)
  assert pole_result.lat == pytest.approx(euler_pole.lat, abs=0)  # 0.002)
  assert pole_result.long == pytest.approx(euler_pole.long)


def test_translation_from_quad():
  gh.setGeod(realWorld=False)

  euler_pole = EulerPole(-90, 45, 0.0)
  v_trans = [1.0, 5.0]  # mm/Yr
  azimuths = [45.0, 135.0, 225.0, 315.0]
  sample_dist = 50  # km

  testPvData = tu.create_simple_sample_quad_w_trans(euler_pole, v_trans, azimuths, sample_dist)

  gn_out = gn.solve_gauss_newton_2D_transform_geo(testPvData, euler_pole.ploc())

  # BUG - The new geo-correct model that (now) works with GPS is pretty bad with quad test. See second version for legacy
  gn_out2 = gn.solve_gauss_newton_translation(testPvData, euler_pole)

  assert gn_out[0] == pytest.approx(v_trans[0])
  assert gn_out[1] == pytest.approx(v_trans[1])


def test_euler_pole_from_random_disk():
  gh.setGeod(realWorld=False)

  euler_pole = EulerPole(-90, 45.0, 1.23)
  sample_count = 400
  diam = 550  # km
  crop = 1.0  # no crop
  test_omega = 1.23

  testPvData = tu.create_random_sample_ring(
          euler_pole, euler_pole.ploc(), sample_count, diam, test_omega, crop)

  pole_result = epr.fit_euler_pole_linear(testPvData)
  # ek.print_result ("test_euler_pole_from_random_disk", pole_result)

  # BUG - The new geo-correct model that (now) works with GPS is pretty bad with quad test. See second version for legacy
  gn_out = gn.solve_gauss_newton_2D_transform_geo(testPvData, pole_result.ploc())
  gn_out2 = gn.solve_gauss_newton_translation(testPvData, pole_result)
  # gn_old = gno.solve_gauss_newton_2D_transform(sample_lons, sample_lats, sample_v_east, sample_v_north, pole_result.ploc())

  assert pole_result.omega == pytest.approx(test_omega)
  assert pole_result.long == pytest.approx(euler_pole.long)
  assert pole_result.lat == pytest.approx(euler_pole.lat)

# show that getting the v-pole from a point and a motion can be reversed


def test_v_pole_from_sample_point():
  gh.setGeod(realWorld=False)

  # typical pnw rot pole setup
  rotPole = EulerPole(-120.1, 44.427, 0.595, is_clockwise=True)
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  v_pavel = PAvel.from_V(v_in)
  pnwVPole = ek.getEulerPoleFromPlocAndPavel(rotPole.ploc(), v_pavel)

  sample_ploc = rotPole.ploc()
  v_out = ek.calculate_v_from_EulerPole(pnwVPole, sample_ploc)
  assert v_out[0] == pytest.approx(v_in[0])
  assert v_out[1] == pytest.approx(v_in[1])


def test_v_pole_from_random_disk():
  gh.setGeod(realWorld=False)

  sample_count = 400
  diam = 550  # km
  crop = 1.0  # no crop
  # typical pnw rot pole setup
  rotPole = EulerPole(-120.1, 44.427, 0.595, is_clockwise=True)
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  v_pavel = PAvel.from_V(v_in)
  pnwVPole = ek.getEulerPoleFromPlocAndPavel(rotPole.ploc(), v_pavel)

  testPvData = tu.create_random_sample_ring(
          pnwVPole, rotPole.ploc(), sample_count, diam, pnwVPole.omega, crop)

  gn_out = gn.solve_gauss_newton_2D_transform_geo(testPvData, rotPole.ploc())
  gn_out2 = gn.solve_gauss_newton_translation(testPvData, rotPole)

  assert gn_out == pytest.approx(v_in, abs=0.005)
  assert gn_out2 == pytest.approx(v_in, abs=0.005)


def test_euler_pole_from_random_cropped_disk():
  gh.setGeod(realWorld=False)

  euler_pole = OC_NA_Pole  # {"lat" : 45.0,  "long" : -90, "omega" : 1.23 }
  sample_count = 400
  sample_dist = 500  # km
  test_omega = 1.23
  crop = 0.5  # 50% cropped out

  testPvData = tu.create_random_sample_ring(
          euler_pole,
          euler_pole.ploc(),
          sample_count,
          sample_dist,
          test_omega,
          crop)
  pole_result = epr.fit_euler_pole_linear(testPvData)
  # ek.print_result ("test_euler_pole_from_random_cropped_disk", pole_result)

  assert pole_result.omega == pytest.approx(test_omega)
  assert pole_result.long == pytest.approx(euler_pole.long)
  assert pole_result.lat == pytest.approx(euler_pole.lat)


def test_euler_pole_using_north_rotation():
  gh.setGeod(realWorld=False)

  euler_pole = OC_NA_Pole
  euler_n_pole = EulerPole(OC_NA_Pole.long + 90, 0.0, 1.43)
  sample_count = 400
  sample_dist = 5000  # km
  test_omega = 1.23
  crop = 0.5  # 50% cropped out
  testPvData = tu.create_random_sample_ring(euler_n_pole,
                                   euler_pole.ploc(),
                                   sample_count,
                                   sample_dist,
                                   test_omega,
                                   crop)

  pole_result = epr.fit_euler_pole_linear(testPvData)
  # ek.print_result ("test_euler_pole_using_north_rotation", pole_result)

  if pole_result.long < 0:
    # euler_n_pole.long +/- 180
    assert pole_result.long == pytest.approx(-29.600000000000488)
  else:
    assert pole_result.long == pytest.approx(180.0 - 29.600000000000488)
  assert pole_result.lat == pytest.approx(euler_n_pole.lat, 0)  # abs=1e-8)
  assert pole_result.omega == pytest.approx(test_omega)

# errors showing up in breaking up rotations

def test_getPoleRotationOfPoint():
  gh.setGeod(realWorld=False)
  test_pole = OC_NA_Pole
  test_loc = SeattlePloc

  target = ek.getPoleRotationOfPoint(test_pole, test_loc, 20.0)
  target_midpoint = ek.getPoleRotationOfPoint(test_pole, test_loc, 10.0)
  target2 = ek.getPoleRotationOfPoint(test_pole, target_midpoint, 10.0)
  assert target == pytest.approx(target2)

def test_euler_GPS_pole_extraction():
  gh.setGeod(realWorld=True)

  center_loc = PLoc(-118.0, 45.0)
  max_distance = 600  # km
  testPvData = tu.get_GPS_rotation_data(center_loc, max_distance)

  pole_result = epr.fit_euler_pole_linear(testPvData)
  # epr.print_result ("test_GPS_pole_extraction", pole_result, len(lats))

  pole_result_sb = EulerPole(-115.406, 43.646, 0.550)
  assert pole_result.long == pytest.approx(pole_result_sb.long, abs=0.001)
  assert pole_result.lat == pytest.approx(pole_result_sb.lat, abs=0.001)
  assert pole_result.omega == pytest.approx(pole_result_sb.omega, abs=0.001)


def test_combined_GPS_pole_extraction():
  gh.setGeod(realWorld=True)

  center_loc = PLoc(-119.0, 45.0)
  max_distance = 600  # km
  testPvData = tu.get_GPS_rotation_data(center_loc, max_distance)

  use_stereo = True
  pole_result, pAvel_result = epr.extractEulerPoleUsingCombinedRegressions(testPvData, use_stereo)
  # epr.print_result ("test_GPS_pole_extraction (use_stereo)", pole_result, len(lats))
  # pAvel_result.print("PAvel_result: ")

  pole_result_sb = EulerPole(-120.162, 44.520, 0.596)
  assert pole_result.long == pytest.approx(pole_result_sb.long, abs=0.001)
  assert pole_result.lat == pytest.approx(pole_result_sb.lat, abs=0.001)
  assert pole_result.omega == pytest.approx(pole_result_sb.omega, abs=0.001)
  paVel_result_sb = PAvel(13.726, 3.700)
  assert pAvel_result.azimuth == pytest.approx(
      paVel_result_sb.azimuth, abs=0.001)
  assert pAvel_result.vel == pytest.approx(paVel_result_sb.vel, abs=0.001)

  use_stereo = False  # use Gemini v regression
  pole_result, pAvel_result = epr.extractEulerPoleUsingCombinedRegressions(testPvData, use_stereo)
  # epr.print_result ("test_GPS_pole_extraction (use Gemini)", pole_result, len(lats))
  # pAvel_result.print("PAvel_result: ")

  pole_result_sb = EulerPole(-119.82, 44.515, 0.596)
  assert pole_result.long == pytest.approx(pole_result_sb.long, abs=0.001)
  assert pole_result.lat == pytest.approx(pole_result_sb.lat, abs=0.001)
  assert pole_result.omega == pytest.approx(pole_result_sb.omega, abs=0.001)
  paVel_result_sb = PAvel(14.748, 3.425)
  assert pAvel_result.azimuth == pytest.approx(
      paVel_result_sb.azimuth, abs=0.001)
  assert pAvel_result.vel == pytest.approx(paVel_result_sb.vel, abs=0.001)


def test_euler_pole_from_pLoc():  # test pnw scenario with northerly motion on pole
  gh.setGeod(realWorld=False)
  ploc = PLoc(OC_NA_Pole.long, OC_NA_Pole.lat)  # sample point loc (arbitrary)
  # point motion north (azimuth, speed in km/ma)
  pAzvel = PAvel(0.0, gh.kmPerDegree())
  pole = ek.getEulerPoleFromPlocAndPavel(ploc, pAzvel)

  # ploc.print("\nploc:")
  # pole.print("pole: ")

  # translation pole 90 degrees off reference at equator
  assert ploc.long - 90 + 360 == pytest.approx(pole.long, abs=1e-6)
  # translation north on meridian has a pone on the equator
  assert pole.lat == pytest.approx(0.0, abs=1e-6)
  assert pole.omega == pytest.approx(1.0, abs=1e-6)


def test_movement_from_Euler_pole():  # test inverse: map above pole back to point
  gh.setGeod(realWorld=False)

  pole = gh.EulerPole(150.4, 0.0, 1.0)
  point = PLoc(OC_NA_Pole.long, OC_NA_Pole.lat)  # sample point loc
  new_point = ek.getPoleRotationOfPoint(pole, point, 1.0)

  # translation pole 90 degrees off reference
  assert new_point.long == pytest.approx(point.long)
  assert new_point.lat == pytest.approx(
      point.lat + 1, abs=1e-6)  # 1 degree shift north
  # distance (km) for 1 degree lat movement
  #assert vel == pytest.approx(gh.kmPerDegree())


def test_v_pole():  # take avg velocity to create pole and then re-create the velocity from the pole
  gh.setGeod(realWorld=False)

  rotPole = EulerPole(-120.1, 44.427, 0.595)  # typocal pnw rot pole setup
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  v_pavel = PAvel.from_V(v_in)

  vPole = ek.getEulerPoleFromPlocAndPavel(rotPole.ploc(), v_pavel)

  v_out = tu.calculate_v_from_Euler_pole2(
      vPole, rotPole.ploc(), vPole.omega, False)
  assert v_out["v_e"] == pytest.approx(v_in[0])
  assert v_out["v_n"] == pytest.approx(v_in[1])

# This is a very old (and naive) interpretation of NA plate motion using NA PAVel, 
# We should update this test (and probably remove or revise ) getPlocFromPoleData
def test_3_pole_50ma_yhs_movement():
  gh.setGeod(realWorld=False)
  yhsLoc0Ma = PLoc(-110.67, 44.43)

  sample_radius = 600  # km
  sample_center = PLoc(-119.0, 45.0)
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  pnwVPole = ek.getEulerPoleFromPlocAndPavel(sample_center, PAvel.from_V(v_in))

  # create samples from pole
  sample_count = 400
  testPvData = tu.create_random_sample_ring(
          pnwVPole, sample_center, sample_count, sample_radius, None)

  # extract translation V from samples
  v_out1 = gn.solve_gauss_newton_2D_transform_geo(testPvData, tu.OC_NA_Pole.ploc())

  # compare Vs
  tolerance1 = 0.006
  assert v_out1 == pytest.approx(v_in, abs=tolerance1)
  pnwRotPole, pnwVPavel = epr.getPnwGpsRotPoleAndVelocity(
      sample_center, sample_radius)

  naPAvel = PAvel(241.0, 23.0)  # degrees, mm / yr
  ma = -50.0

  yhsLoc50Ma = ek.getPlocFromPoleData(
      naPAvel, pnwRotPole, pnwVPavel, yhsLoc0Ma, ma)

  ploc50maSB = PLoc(-124.346, 41.522)
  assert yhsLoc50Ma.lat == pytest.approx(ploc50maSB.lat, abs=0.001)
  assert yhsLoc50Ma.long == pytest.approx(ploc50maSB.long, abs=0.001)


def angle_difference(a, b):
    # Returns the shortest angular distance in degrees [-180, 180]
  return (a - b + 180) % 360 - 180

# Creates a synthethetic rot field (based on Wells-Smith OC_NA pole) and extracts that pole back
def test_rot_regressions_against_sim_data():
  gh.setGeod(realWorld=False)
  rotPole = tu.OC_NA_Pole

  # create samples from pole
  sample_count = 400
  testPvData = tu.create_random_sample_ring(
          rotPole, tu.sample_center, sample_count, tu.sample_radius, None)

  # extract rot pole from samples
  out_pole = epr.fit_euler_pole_linear(testPvData)

  # compare poles
  assert out_pole.long == pytest.approx(rotPole.long)
  assert out_pole.lat == pytest.approx(rotPole.lat)
  assert out_pole.omega == pytest.approx(rotPole.omega)

def test_v_regression_against_sim_data():
  gh.setGeod(realWorld=False)

  # pole for generating samples based on V
  # sample_center = PLoc(-119.0, 45.0)

  sample_center = tu.OC_NA_Pole.ploc()  # same as combined pole setup
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  inPavel = PAvel.from_V(v_in)
  pnwVPole = ek.getEulerPoleFromPlocAndPavel(sample_center, inPavel)

  # create samples from pole
  sample_count = 400
  testPvData = tu.create_random_sample_ring(
          pnwVPole, sample_center, sample_count, tu.sample_radius, None)

  # extract translation V from samples
  v_out = gn.solve_gauss_newton_2D_transform_geo(testPvData, tu.OC_NA_Pole.ploc())
  outPavel = PAvel.from_V(v_out)

  # compare
  az_diff = angle_difference(inPavel.azimuth, outPavel.azimuth)
  assert az_diff == pytest.approx(0, abs=0.0125)
  assert outPavel.vel == pytest.approx(inPavel.vel, abs=0.01)

# One test to validate the combined pole simulations result in a net zero velocity
def test_compound_pole_v():
  gh.setGeod(realWorld=False)

  # pole for generating samples based on omega
  rotPole = tu.OC_NA_Pole

  # pole for generating samples based on V
  sample_center = rotPole.ploc()  # the configuration for coupled poles
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  inVPavel = PAvel.from_V(v_in)
  vPole = ek.getEulerPoleFromPlocAndPavel(sample_center, inVPavel)

  distance_to_match = tu.distance_for_target_velocity(
      rotPole.omega, inVPavel.vel)

  loc_match = gh.create_sample(
      sample_center.long, sample_center.lat, inVPavel.azimuth + 90.0, distance_to_match)

  v1 = ek.calculate_v_from_EulerPole(vPole.normalize(), loc_match)
  v2 = ek.calculate_v_from_EulerPole(rotPole.normalize(), loc_match)

  assert v1 == pytest.approx(-v2, abs=0.002)


def test_combined_regressions_against_sim_data():
  gh.setGeod(realWorld=False)

  rotPole = tu.OC_NA_Pole

  # pole for generating samples based on V
  sample_center = rotPole.ploc()  # the configuration for coupled poles
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  inVPavel = PAvel.from_V(v_in)
  vPole = ek.getEulerPoleFromPlocAndPavel(sample_center, inVPavel)

  # create samples from pole
  sample_count = 400
  testPvData = tu.create_random_sample_dual_pole_ring(
          vPole, rotPole, sample_center, sample_count, tu.sample_radius)

  # extract translation V from samples
  out_rot_pole, out_vPavel = epr.extractEulerPoleUsingCombinedRegressions(testPvData)
  # print(f"out_v: {out_vPavel}")

  # compare poles
  az_diff = angle_difference(inVPavel.azimuth, out_vPavel.azimuth)
  assert az_diff == pytest.approx(0, abs=4.1)
  assert out_vPavel.vel == pytest.approx(inVPavel.vel, abs=0.105)

  assert out_rot_pole.long == pytest.approx(rotPole.long, abs=0.02)
  assert out_rot_pole.lat == pytest.approx(rotPole.lat, abs=0.11)
  assert out_rot_pole.omega == pytest.approx(rotPole.omega, abs=0.006)
