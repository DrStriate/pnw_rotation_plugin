import test_utils as tu
import numpy as np
from .src.geo_helper import PLoc, PVel
import euler_pole_regression as epr
import gauss_newton as gn

delta_ve = 0.0
delta_vn = 0.0


def clear_test_run_pass():
  global delta_ve, delta_vn
  delta_ve = 0.0
  delta_vn = 0.0


def run_GPS_test_pass(self):
  global delta_ve, delta_vn
  # get rot data
  diam = 600  # km
  center_loc = PLoc(-119.0, 45.0)
  lat_list, long_list, ve_list, vn_list, se, sn =\
      tu.get_GPS_rotation_data(center_loc, diam * 1000)
  mod_ve_list = np.array(ve_list) - delta_ve
  mod_vn_list = np.array(vn_list) - delta_vn
  delta_ve, delta_vn = finish_test_setup(
      self, lat_list, long_list, mod_ve_list, mod_vn_list, diam, delta_ve, delta_vn, se, sn)


def run_quad_test_pass(self):
  # get rot data
  euler_pole = tu.OC_NA_Pole  # {"lat" : 45.0,  "long" : -90, "omega" : 1.23 }
  azimuths = [45.0, 135.0, 225.0, 315.0]
  diam = 50  # km
  long_list, lat_list, ve_list, vn_list = tu.create_simple_sample_quad(
      euler_pole, azimuths, diam * 1000)
  finish_test_setup(self, lat_list, long_list, ve_list, vn_list, diam)


def run_rand_disk_test_pass(self):
  # get rot data
  euler_pole = tu.OC_NA_Pole  # {"lat" : 45.0,  "long" : -90, "omega" : 1.23 }
  sample_count = 400
  diam = 400  # km
  crop = 1.0  # no crop
  test_omega = 1.23

  lat_list, long_list, ve_list, vn_list = \
      tu.create_random_sample_ring(
          euler_pole, euler_pole.ploc(), sample_count, diam * 1000, test_omega, crop)
  finish_test_setup(self, lat_list, long_list, ve_list, vn_list, diam)


def run_cropped_disk_test_test_pass(self):

  euler_pole = tu.OC_NA_Pole  # {"lat" : 45.0,  "long" : -90, "omega" : 1.23 }
  sample_count = 400
  diam = 400  # km
  test_omega = 1.23
  crop = 0.5  # 50% cropped out

  lat_list, long_list, ve_list, vn_list = \
      tu.create_random_sample_ring(
          euler_pole,
          euler_pole.ploc(),
          sample_count,
          diam * 1000,
          test_omega,
          crop,
          None)
  finish_test_setup(self, lat_list, long_list, ve_list, vn_list, diam)


def finish_test_setup(self, lat_list, long_list, ve_list, vn_list, diam, d_ve=None, d_vn=None, s_e=None, s_n=None):
  # clear and set rot data in Qgis
  self.rotDestLayer.dataProvider().truncate()
  self.yhsRotFeatureList = []
  for i in range(len(lat_list)):
    feature = self.rotData.createRotFeature(
        PLoc(long_list[i], lat_list[i]), PVel(ve_list[i], vn_list[i]))
    self.yhsRotFeatureList.append(feature)

  # get euler pole and gauss newton results and display
  pole = epr.fit_euler_pole_linear(
      lat_list, long_list, ve_list, vn_list, s_e, s_n)
  gn_out = gn.solve_gauss_newton_2D_transform_geo_wtd(
      long_list, lat_list, ve_list, vn_list, s_e, s_n, pole)

  label_text1 = f"{pole.long:.4f}, {pole.lat:.4f}, {pole.omega:.3f} deg, "
  label_text2 = f"e: {(gn_out['t_x'] / 1E3):.2f} km, n: {(gn_out['t_y'] / 1E3):.2f} km, {diam} km"
  self.geoWhiteboard.draw_target(
      pole.long, pole.lat, label_text1 + label_text2)
  # print(label_text1 + label_text2)

  # if translation correction added, add a delta_V vector to the target to sho that
  if d_ve != 0.0 or d_vn != 0.0:
    feature = self.rotData.createRotFeature(
        PLoc(pole.long, pole.long), PVel(delta_ve + d_ve, delta_vn + d_vn))
    self.yhsRotFeatureList.append(feature)

  return gn_out['t_x'], gn_out['t_y']
