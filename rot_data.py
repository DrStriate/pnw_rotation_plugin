from qgis._core import QgsMessageLog, Qgis, QgsProject, QgsVectorLayer
from qgis.core import QgsFeature, QgsPointXY, QgsGeometry
from qgis.PyQt.QtCore import QVariant, QDateTime, Qt

import os
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import geopandas as gpd
from .src import geo_helper as gh


@dataclass
class PState:
  longitude: float
  latitude: float
  vEast: float
  vNorth: float


class RotData:
  XRange = {"min": -125.0, "max": -110.0}
  YRange = {"min": 37.0, "max": 50.0}
  XYSteps = 50
  XSize = (XRange["max"] - XRange["min"]) / XYSteps
  YSize = (YRange["max"] - YRange["min"]) / XYSteps

  # RotEntryExclusions = [] # no outliers taken out
  RotEntryExclusions = [438, 3659]  # clear outliers
  # RotEntryExclusions = [2425, 438, 3659] # more consistent final path

  def __init__(self):
    self.rotSourceLayer = None
    self.rotSourceFields = None
    self.rotDataLoaded = False
    self.rotFieldList = []
    self.rotFeatureList = []
    self.u_interp = None
    self.v_interp = None
    self.interp_func = None
    self.interpFunction = "ClosestEntry"
    self.gdf = None
    self.rotDataLayerName = 'NSHM2023_GPS_velocity'

  def loadFromFile(self):
    gpsFIlePath = 'data/NSHM2023_GPS_velocity.zip'
    self.gdf = gpd.read_file(f"zip://{gpsFIlePath}")
    return

  def clearData(self):
    self.rotFieldList = []
    self.rotFeatureList = []
    return

  def load(self):
    if self.rotDataLoaded:
      return True

    # 1. Get the absolute path of your plugin directory
    plugin_dir = os.path.dirname(__file__)

    # 2. Point to the zip file (and optionally the specific .shp inside it if there are multiple files)
    zip_file_path = os.path.join(plugin_dir, "NSHM2023_GPS_velocity.zip")

    # Format the GDAL virtual zip URI
    # If the zip has a single shapefile at its root, just pointing to the zip works:
    uri = f"/vsizip/{zip_file_path}"

    # gpsFIlePath = 'data/NSHM2023_GPS_velocity.zip'
    self.rotSourceLayer = QgsVectorLayer(
        uri,
        self.rotDataLayerName,
        "ogr"
    )

    QgsProject.instance().addMapLayer(self.rotSourceLayer)

    node = QgsProject.instance().layerTreeRoot().findLayer(self.rotSourceLayer.id())
    node.setItemVisibilityChecked(False)

    # Path to your style file inside the plugin folder
    style_path = os.path.join(plugin_dir, "velocity_style.qml")

    # Apply the style to the layer
    success, message, = self.rotSourceLayer.loadNamedStyle(style_path)

    if success:
      print(self.rotSourceLayer.source())

      # get rot data fields
      self.rotSourceFields = self.rotSourceLayer.fields()
      for field in self.rotSourceFields:
        self.rotFieldList.append(field)

      # get rot features
      featureList = self.rotSourceLayer.getFeatures()
      for feature in featureList:
        self.rotFeatureList.append(feature)

      self.rotDataLoaded = True

    else:
      self.rotDataLoaded = False
      print(f"Failed to load style: {message}")

    return self.rotDataLoaded

  def closeRotSourceLayer(self):
    # Check if the layer exists and is registered in the project
    if self.rotSourceLayer:
      # Remove it from the map registry (this closes/deletes it from the layers panel)
      QgsProject.instance().removeMapLayer(self.rotSourceLayer.id())
      self.rotSourceLayer = None
    self.rotDataLoaded = False

  def createRotFeature(self, pLoc, pdist, d_scaling=1.0):
    # --- Create a new QgsFeature instance ---
    new_feature = QgsFeature(self.rotSourceFields)
    geometry = QgsGeometry.fromPointXY(QgsPointXY(pLoc.long, pLoc.lat))

    # Assign the created geometry to the feature
    new_feature.setGeometry(geometry)

    # --- Set the Attributes ---
    # Provide a list of values that match the field order defined above
    attributes = [pLoc.long, pLoc.lat, pdist.east *
                  d_scaling, pdist.north * d_scaling]

    # Assign the attributes to the feature
    new_feature.setAttributes(attributes)

    return new_feature

  # Setup routines to prep for sampling depending on interpFunction
  def setupSampling(self, interpFunction):
    self.interp_func = interpFunction
    if self.interp_func == "LinearNDInterpolator":
      self.linearInterpSamplerSetup()
    return

  # Sample functions dependent on interp_function setting
  def sample(self, longitude, latitude):
    if self.interpFunction == "ClosestEntry":
      return self.getClosestRotEntry(longitude, latitude)
    elif self.interpFunction == "LinearNDInterpolator":
      return self.getLinearInterpSample(longitude, latitude)
    return None

  def linearInterpSamplerSetup(self):
    # build x (longitude), y (latitude), u (Ve) and v (vn) arrays
    x, y, u, v = [], [], [], []
    feature_idx = 0
    for feature in self.rotFeatureList:
      if feature_idx not in self.RotEntryExclusions:
        x.append(feature.attribute(0))
        y.append(feature.attribute(1))
        u.append(feature.attribute(2))
        v.append(feature.attribute(3))
      feature_idx += 1

    points = np.column_stack((x, y))  # (N, 2)
    values = np.column_stack((u, v))  # (N, 2)
    if len(points) > 0:
      self.interp_func = LinearNDInterpolator(points, values)
    return

  def getLinearInterpSample(self, longitude, latitude):
    if longitude < self.XRange["min"] or longitude > self.XRange["max"]:
      print("Longitude out of range: " + f"{longitude:.4f}" + ". Clamping.")
      longitude = gh.clamp(longitude, self.XRange["min"], self.XRange["max"])
    if latitude < self.YRange["min"] or latitude > self.YRange["max"]:
      print("latitude out of range: " + f"{latitude:.4f}" + ". Clamping.")
      latitude = gh.clamp(latitude, self.YRange["min"], self.YRange["max"])

    u, v = self.interp_func(longitude, latitude)
    return PState(longitude, latitude, u, v)

  # returns PState of closest rot vector
  def getClosestRotEntry(self, longitude, latitude):
    if not self.rotDataLoaded:
      if not self.load():
        QgsMessageLog.logMessage(
            'No rotation data loaded', RotData.name, Qgis.Info)
        return None

    closestIdx = -1
    minDist = 1e10
    idx = 0
    for feature in self.rotFeatureList:
      f_lon = feature.attribute(0)
      f_lat = feature.attribute(1)
      dist = (f_lon - longitude)**2 + (f_lat - latitude)**2
      if dist < minDist and idx not in self.RotEntryExclusions:
        minDist = dist
        closestIdx = idx
      idx += 1

    if closestIdx == -1:
      return None

    # self.compareRotFeatureToInterp(self.rotFeatureList[closestIdx])

    return PState(
        self.rotFeatureList[closestIdx].attribute(0),
        self.rotFeatureList[closestIdx].attribute(1),
        self.rotFeatureList[closestIdx].attribute(2),
        self.rotFeatureList[closestIdx].attribute(3))

  def getRotationV(self, longitude, latitude, interpolate):
    if (not interpolate):
      # get the closest rotation entry velocity for current location
      rotEntry = self.getClosestRotEntry(longitude, latitude)
    else:  # Interpolated
      rotEntry = self.getLinearInterpSample(longitude, latitude)
    if rotEntry is None or np.isnan(rotEntry.vEast) or np.isnan(rotEntry.vNorth):
      print('No close rotation entry found')
      return [0, 0], False
    return [rotEntry.vEast / 1000.0, rotEntry.vNorth / 1000.0], True  # m / yr

  def interpTest(self):
    x, y, u, v = [], [], [], []
    # for feature in self.rotFeatureList:
    #     compareRotFeatureToInterp(self, feature)
    # x.append(lon)
    # y.append(lat)
    # u.append(veErr)
    # v.append(vnErr)

    # plt.quiver(x, y, u, v)
    # plt.show()
    return

  def compareRotFeatureToInterp(self, feature):
    lon = feature.attribute(0)
    lat = feature.attribute(1)
    ve = feature.attribute(2)
    vn = feature.attribute(3)

    interpState = self.getInterpRotEntry(lon, lat)
    if interpState is not None:
      lonErr = lon - interpState.longitude
      latErr = lat - interpState.latitude
      veErr = ve - interpState.vEast
      vnErr = vn - interpState.vNorth

      print(
          "veErr: " + str(veErr) + " (" + str(ve) + " raw vs " + str(interpState.vEast) + " interp)\n" +
          "vnErr: " + str(vnErr) + " (" + str(vn) + " raw vs " + str(interpState.vNorth) + " interp)")
    else:
      print("No interp found")
    return
