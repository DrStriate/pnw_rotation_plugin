import numpy as np
import pytest
import gauss_newton as gn
import test_utils as tu
from pathlib import Path
from geo_helper import PLoc, PAvel, EulerPole, geod, R
import geo_helper as gh
import euler_kinematics as ek

# center at lat 512, long 512. Distance to center is 128
OC_NA_Pole = {"lat": 45.54,  "long": -119.60, "omega": 1.32}

sample_e = [512, 384, 512, 640]
sample_n = [384, 512, 640, 512]


def test_translation_east():
  # test 1 = translate east
  t_e = 1.0
  v_e = [t_e, t_e, t_e, t_e]
  v_n = [0.0, 0.0, 0.0, 0.0]
  x = gn.solve_gauss_newton_2D_transform(sample_e, sample_n, v_e, v_n)
  # print(f"east translate X: {x}\n")
  assert x[0] == t_e
  assert x[1] == 0.0


def test_translation_north():
  # test 2 = translate north
  t_n = 1.0
  v_e = [0.0, 0.0, 0.0, 0.0]
  v_n = [t_n, t_n, t_n, t_n]
  x = gn.solve_gauss_newton_2D_transform(sample_e, sample_n, v_e, v_n)
  # print(f"North translate X: {x}\n")
  assert x[0] == 0.0
  assert x[1] == t_n


def test_gm_regress_against_pnw_GPS_data():
  # Absolute path of the script
  script_path = Path(__file__).resolve()
  # print(script_path)

  euler_pole = EulerPole(-118.5, 45, 0)
  center_ploc = euler_pole.ploc()
  max_distance = 550  # m

  testPvData = tu.get_GPS_rotation_data(center_ploc, max_distance)

  x = gn.solve_gauss_newton_translation(testPvData, euler_pole)

  # print(f"samples: {len(lats)}")
  # gn.print_x(x)

  x_sb = np.array([1.215, 3.164])
  assert x == pytest.approx(x_sb, abs=0.001)

# Tests legacy 2D imaging-based gauss-newton compared to the Gemini variant)
def test_gn_regressions_against_sim_data():
  gh.setGeod(realWorld=False)

  # pole for generating samples based on V
  sample_center = PLoc(-119.0, 45.0)
  v_in = [0.767, 3.545]  # v pavel from typical calibration
  pnwVPole = ek.getEulerPoleFromPlocAndPavel(sample_center, PAvel.from_V(v_in))

  # create samples from pole
  sample_count = 400
  testPvData = tu.create_random_sample_ring(
          pnwVPole, tu.sample_center, sample_count, tu.sample_radius, None)

  # extract translation V from samples
  v_out1 = gn.solve_gauss_newton_2D_transform_geo(testPvData, tu.OC_NA_Pole.ploc())

  # compare Vs
  tolerance1 = 0.006
  assert v_out1 == pytest.approx(v_in, abs=tolerance1)

  # extract translation V from samples using Gemini regression (need a full pole to make it work)
  input_pole = EulerPole(sample_center.long, sample_center.lat, 0.6)
  v_out2 = gn.solve_gauss_newton_translation(testPvData, input_pole)

  # compare Vs
  tolerance2 = 0.061
  assert v_out2 == pytest.approx(v_in, abs=tolerance2)
