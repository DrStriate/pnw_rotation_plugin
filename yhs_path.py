import numpy as np
from dataclasses import dataclass, asdict, replace
from .src import euler_pole_regression as epr
import json
import dacite
import os

from .src.geo_helper import PAvel, PLoc, PVel, EulerPole, pvData, getPAvel, setGeod
from .src import test_utils as tu
from .src import euler_pole_regression as epr
from .src import gauss_newton as gn
from .src import euler_kinematics as ek
from . import path_layer as pl
from .na_plate_gplates import read_yhs_location as ry

YHS_lat = 44.43           # Yellowstone caldera yhs @ 0 Ma
YHS_long = -110.67

# NA_plate_speed = 23.0     # mm / yr (Current) = Adjusted to Owyhee=Humbolt cauldera ~14Ma
# NA_plate_azimuth = 241.0  # degrees

@dataclass
class YhsPropertyBag:
  NaPlateDataName: str
  PnwVPAvel: PAvel
  PnwRotPole: EulerPole
  StereographicProjection: bool

class YhsPath:
  def __init__(self, parent):
    self.parent = parent
    self.path_layer_manager = pl.PathLayerManager()

    # Set up deault plate data
    self.setupNAPLateData("yhs_continuous_1ma_Muller2019.geojson")
    self.useGpsData = True
    self.pole_model = 2  # 1 is NA then Pole-V then Pole-R, 2 is NA - Translated-R - Pole-V
    self.yhs_loc = PLoc(YHS_long, YHS_lat)

    # parameters for PWN rot sample data
    self.sample_radius = 600  # km
    self.sample_center = PLoc(-119.0, 45.0)

    self.stereographicProjection = True

    self.delta_ve = 0
    self.delta_vn = 0

    # pre-load pnw GPS data set
    self.pvData = tu.get_GPS_rotation_data(self.sample_center, self.sample_radius)
    self.sample_count = len(self.pvData.lats)

  # Serialization (Object -> JSON)
  def get_serialize_bag(self) -> str:
    # asdict recursively handles nesting automatically
    return json.dumps(asdict(self.getYhsPropertyBag()), indent=4)

  # Deserialization (JSON -> Object)
  def deserialize_and_set_bag(self, json_str: str) -> YhsPropertyBag:
    data_dict = json.loads(json_str)

    propertyBag = dacite.from_dict(data_class=YhsPropertyBag, data=data_dict)
    self.NaPlateDataName = propertyBag.NaPlateDataName
    self.PnwVPAvel = propertyBag.PnwVPAvel
    self.PnwRotPole = propertyBag.PnwRotPole
    self.stereographicProjection = propertyBag.StereographicProjection

  def getYhsPropertyBag(self):
    propertyBag = YhsPropertyBag(
        self.NaPlateDataName, self.PnwVPAvel, self.PnwRotPole, self.stereographicProjection)
    return propertyBag

  def setYhsPropertyBag(self, propertyBag):
    self.NaPlateDataName = propertyBag.NaPlateDataName
    self.PnwVPAvel = propertyBag.PnwVPAvel
    self.PnwRotPole = propertyBag.PnwRotPole
    self.stereographicProjection = propertyBag.StereographicProjection

  def setupNAPLateData(self, fileName):
    script_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "na_plate_gplates")
    na_file_path = os.path.join(script_dir, fileName)
    self.yhs_path_data = ry.load_data(na_file_path)
    self.NaPlateDataName = fileName

  def getPnwGpsRotPoleAndVelocity(self):
    self.PnwRotPole, self.PnwVPAvel = epr.getPnwGpsRotPoleAndVelocity2(
        self.pvData, self.stereographicProjection)

  def checkLayersCreated(self):
    self.NaPoleLayer = self.path_layer_manager.getInstance("NA pole", "red")
    self.PnwVPoleLayer = self.path_layer_manager.getInstance(
        "Pnw V pole ", "blue")
    self.PnwRotPoleLayer = self.path_layer_manager.getInstance(
        "Pnw Rot pole", "black")
    self.PnwComboLayer = self.path_layer_manager.getInstance(
        "Pnw R-V pole", "green")

  # any statefulness that changes with runs should be resettable
  def erase_everything(self):
    if self.path_layer_manager is not None:
      self.path_layer_manager.erase_everything()

  def closeLayers(self):
    self.path_layer_manager.close_layers()

  def modeSet(self, poleModel):
    self.pole_model = poleModel

  # Plot and label the NA Euler rotation pole
  def display_NA_pole_info(self):
    label_text1 = f"0 Ma YHS ({self.yhs_loc.long:.3f}, {self.yhs_loc.lat:.3f}), "
    # label_text2 = f"az: {self.NAPAvel.azimuth:.1f} deg, v: {(self.NAPAvel.vel):.1f} km/Ma"
    self.parent.geoWhiteboard.draw_target(
        self.yhs_loc.long, self.yhs_loc.lat, label_text1)
    self.parent.geoWhiteboard.draw_target(self.PnwRotPole.long, self.PnwRotPole.lat,
                                          f"0 Ma pole ({self.PnwRotPole.long:0.3f}, {self.PnwRotPole.lat:0.3f})")

  def get_yhs_loc(self, currentMa, deltaMa, steps):  # yrs is years
    self.checkLayersCreated()

    # get a local rot pole we can move as needed
    runPnwRotPole = replace(self.PnwRotPole)
    # runPnwRotPole.print("runPnwRotPole: ") # these printouts useful for setting up GPlates

    # get the PnwVPole which is time invariant
    pnwVPole = ek.getEulerPoleFromPlocAndPavel(runPnwRotPole.ploc(), self.PnwVPAvel)
    # pnwVPole.print("PnwVPole: ")

    # 1: Move yhs loc by NA speed scaled by ma from 0 Ma location (red line)
    # self.NAPole = ek.getEulerPoleFromPlocAndPavel(self.yhs_loc, self.NAPAvel)
    # loc_1 = self.NaPoleLayer.RenderPoleMotionForMa(self.yhs_loc, self.NAPole, -currentMa)
    loc_1 = self.NaPoleLayer.RenderYHSPoleMotionForMa(
        self.yhs_path_data, -currentMa)
    # loc_1.print("loc_1")

    # 'R-V' mode: move YHS loc by PnwVPole then rotate (most direct model) 
    if self.pole_model == 1:

      # 2: Move by ma scaled pole translation v (blue line) - note pole is position dpendent
      loc_2 = self.PnwVPoleLayer.RenderPoleMotionForMa(
          loc_1, pnwVPole, -currentMa)
      # loc_2.print("loc_2: ")

      # 3: Rotate by ma scaled pole omega
      loc_3 = self.PnwRotPoleLayer.RenderPoleMotionForMa(
          loc_2, runPnwRotPole, currentMa)
      self.parent.geoWhiteboard.draw_target(
          loc_3.long, loc_3.lat, f"{currentMa} Ma YHS ({loc_3.long:0.3f}, {loc_3.lat:0.3f})")
      # loc_3.print("loc_3: ")

    # 'V-R' mode: move rot pole down by pole_v, rotate loc_1 to loc_2
    elif self.pole_model == 2: 
      # 2: Translate rot pole
      new_rot_pole_ploc = ek.getPoleRotationOfPoint(
          pnwVPole, runPnwRotPole.ploc(), currentMa)
      t_RotPole = EulerPole(
          new_rot_pole_ploc.long, new_rot_pole_ploc.lat, runPnwRotPole.omega, is_clockwise=True)
      self.parent.geoWhiteboard.draw_target(t_RotPole.long, t_RotPole.lat,
                                            f"{currentMa} Ma pole ({t_RotPole.long:0.3f}, {t_RotPole.lat:0.3f})")
      # Rotate loc by pre-translated rot pole
      loc_2 = self.PnwRotPoleLayer.RenderPoleMotionForMa(
          loc_1, t_RotPole, currentMa)  # WHY NOW POSITIVE Ma?
      # loc_2.print("loc_2: ")

      # 3: Translate back up
      loc_3 = self.PnwVPoleLayer.RenderPoleMotionForMa(
          loc_2, pnwVPole, -currentMa)
      self.parent.geoWhiteboard.draw_target(
          loc_3.long, loc_3.lat, f"{currentMa} Ma YHS ({loc_3.long:0.3f}, {loc_3.lat:0.3f})")
      # loc_3.print("loc_3: ")

    # 'Combo' mode, model 3: run pole from start Ma but plot progress points up to final ma
    elif self.pole_model == 3: 
      # self.parent.geoWhiteboard.draw_target(loc_1.long, loc_1.lat, f"{currentMa} Ma YHS ({loc_1.long:0.3f}, {loc_1.lat:0.3f})")
      loc_3 = self.PnwComboLayer.RenderComboPoleMotionForMa(
          loc_1, pnwVPole, runPnwRotPole, currentMa)
      self.parent.geoWhiteboard.draw_target(
          loc_3.long, loc_3.lat, f"{currentMa} Ma YHS ({loc_3.long:0.3f}, {loc_3.lat:0.3f})")
      self.PnwRotPoleLayer.RenderAzimuthMarkersforMa(
          loc_1, pnwVPole, runPnwRotPole, currentMa)

    # 'YHS Only' mode, model 4: plot only the final YHS spot
    else:  
      # self.parent.geoWhiteboard.draw_target(loc_1.long, loc_1.lat, f"{currentMa} Ma YHS ({loc_1.long:0.3f}, {loc_1.lat:0.3f})")
      loc_3 = ek.getCompoundRotationTranslationOfPoint(
          pnwVPole, runPnwRotPole, loc_1, currentMa)
      self.parent.geoWhiteboard.draw_target(
          loc_3.long, loc_3.lat, f"{currentMa} Ma YHS ({loc_3.long:0.3f}, {loc_3.lat:0.3f})")

    return loc_3

  def displayGPSDataAndPoles(self, test_code):  
    setGeod(realWorld=True)

    off_pvData = None
    if test_code == 1:  # show synthetic v data
      # pole for generating samples
      v_in = [0.767, 3.545]  # v pavel from typical calibration
      v_pavel = PAvel.from_V(v_in)
      pnwVPole = ek.getEulerPoleFromPlocAndPavel(self.sample_center, v_pavel)

      crop = 1.0  # no crop
      pv_data = tu.create_random_sample_ring(
              pnwVPole, self.sample_center, self.sample_count, self.sample_radius * 1000.0, pnwVPole.omega, crop)

    elif test_code == 2:  # show synthetic rot data
      # pole for generating samples
      rotPole = tu.OC_NA_Pole

      # create samples from pole
      pv_data = tu.create_random_sample_ring(
              rotPole, tu.sample_center, self.sample_count, tu.sample_radius, None)

    elif test_code == 3:
      # pole for generating samples based on V
      sample_center = PLoc(-119.0, 45.0)
      v_in = [0.767, 3.545]  # v pavel from typical calibration
      vPole = ek.getEulerPoleFromPlocAndPavel(
          sample_center, PAvel.from_V(v_in))
      rotPole = tu.OC_NA_Pole

      # create samples from pole
      pv_data = tu.create_random_sample_dual_pole_ring(
              vPole, rotPole, tu.sample_center, self.sample_count, tu.sample_radius, None)

    else:  # show and analyze velocity data
      if self.useGpsData:
        pv_data = tu.get_GPS_rotation_data(self.sample_center, self.sample_radius)
      else:
        pnwVPole = ek.getEulerPoleFromPlocAndPavel(self.PnwRotPole.ploc(), self.PnwVPAvel)
        pv_data =  tu.create_random_sample_dual_pole_ring(
              pnwVPole, self.PnwRotPole, self.PnwRotPole.ploc(), self.sample_count, self.sample_radius)

      if len(pv_data.lats) < 3:
        return False

      # offset_data by detected v
      off_pvData = replace(pv_data, v_es = np.array(pv_data.v_es) - self.delta_ve, v_ns = np.array(pv_data.v_ns) - self.delta_vn)

      # get euler pole and gauss newton results and display
      self.PnwRotPole = epr.fit_euler_pole_linear(off_pvData)
      self.PnwRotPole.print("self.PnwRotPole: ")

      offsets = gn.solve_gauss_newton_transform(off_pvData, self.PnwRotPole, self.stereographicProjection)
      print(f"offsets: {offsets}")

      label_text1 = f"{self.PnwRotPole.long:.4f}, {self.PnwRotPole.lat:.4f}, {self.PnwRotPole.omega:.3f} deg, "
      label_text2 = f"e: {offsets[0]:.3f} km, n: {offsets[1]:.3f} km, {self.sample_radius} km"
      self.parent.geoWhiteboard.draw_target(
          self.PnwRotPole.long, self.PnwRotPole.lat, label_text1 + label_text2)
      print("Label_text1: " + label_text1 + label_text2)

      # if translation correction added, add a delta_V vector to the target to show that
      if self.delta_ve != 0.0 or self.delta_vn != 0.0:
        feature = self.parent.rotData.createRotFeature(
            PLoc(self.PnwRotPole.long, self.PnwRotPole.lat), PVel(self.delta_ve + offsets[0], self.delta_vn + offsets[1]))
        self.parent.yhsRotFeatureList.append(feature)
        self.PnwVPAvel = getPAvel(self.delta_ve, self.delta_vn)
      else:
        self.PnwVPAvel = PAvel(0, 0)

      self.delta_ve = offsets[0]
      self.delta_vn = offsets[1]

    self.parent.rotDestLayer.dataProvider().truncate()
    self.parent.yhsRotFeatureList = []
    if off_pvData is None:
      off_pvData = pv_data
    for i in range(len(pv_data.lats)):
      feature = self.parent.rotData.createRotFeature(
          PLoc(pv_data.longs[i], pv_data.lats[i]), PVel(off_pvData.v_es[i], off_pvData.v_ns[i]))
      self.parent.yhsRotFeatureList.append(feature)
    return True

  def clearGPSparams(self):
    self.delta_ve = 0.0
    self.delta_vn = 0.0
