from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField
)
from PyQt5.QtCore import QMetaType
from PyQt5.QtGui import QColor

import numpy as np
from .src import euler_kinematics as ek
from .src import geo_helper as gh
from .na_plate_gplates import read_yhs_location as ry


class PathLayerManager():
  def __init__(self):
    self.layers = []

  def getInstance(self, name, color="red"):
    instance = next(
        (item for item in self.layers if item.layer_name == name), None)
    if instance is None:
      instance = PathLayer(name, color)
      self.layers.append(instance)
    return instance

  def erase_everything(self):
    for layer in self.layers:
      layer.clear_layer()

  def close_layers(self):
    for layer in self.layers:
      layer.unload()
    self.layers = []


class PathLayer():
  def __init__(self, path_layer_name, color="red"):
    self.layer_name = path_layer_name

    # 1. Define the memory layer URI
    # "LineString" defines the geometry type.
    # "crs=epsg:4326" sets it to standard WGS 84 (Lat/Lon).
    uri = "LineString?crs=epsg:4326"

    # 2. Create the vector layer
    self.qvector_layer = QgsVectorLayer(uri, self.layer_name, "memory")

    # 3. Add attributes (optional, but useful if you want to label or color them later)
    provider = self.qvector_layer.dataProvider()
    provider.addAttributes([QgsField("path_name", QMetaType.Type.QString)])
    self.qvector_layer.updateFields()  # Tell the layer to recognize the new fields

    renderer = self.qvector_layer.renderer()
    symbol = renderer.symbol()
    symbol.setColor(QColor(color))

    # Set the width (in millimeters by default in QGIS)
    symbol.setWidth(0.5)

  def add_run_paths_to_path_layer(self, raw_paths):
    features = []
    for start, end, name in raw_paths:
      # Create a new feature
      feature = QgsFeature()

      # Create the line geometry from start and end points
      # Note: QgsPointXY takes (Longitude/X, Latitude/Y)
      start_point = QgsPointXY(start[0], start[1])
      end_point = QgsPointXY(end[0], end[1])

      line_geom = QgsGeometry.fromPolylineXY([start_point, end_point])
      feature.setGeometry(line_geom)

      # Add attribute values matching the fields we created
      feature.setAttributes([name])
      features.append(feature)

    # 5. Write the features to the layer
    provider = self.qvector_layer.dataProvider()
    provider.addFeatures(features)

    # Update the layer's extents so QGIS knows how big it is
    self.qvector_layer.updateExtents()

    # 6. Add the layer to the current QGIS Project so it draws on screen
    QgsProject.instance().addMapLayer(self.qvector_layer)

    self.qvector_layer.triggerRepaint()

  # plots path from start_point around Euler pole for specified time (ma)
  def RenderPoleMotionForMa(self, start_point, pole, ma, N=20):
    layer_paths = []
    start_loc = start_point
    for i in range(0, N+1):
      next_ma = i * ma / N
      next_loc = ek.getPoleRotationOfPoint(pole, start_point, next_ma)
      layer_paths.append(
          [(start_loc.long, start_loc.lat), (next_loc.long, next_loc.lat), f"rot step {N}"])
      start_loc = next_loc
      self.add_run_paths_to_path_layer(layer_paths)
      # print(f"next_ma {next_ma}")
    return next_loc

  def RenderYHSPoleMotionForMa(self, yhs_path_data, ma):
    layer_paths = []
    # lat, long = ry.get_interpolated_yhs_position(yhs_path_data, 0.0)
    t1, lat, long = yhs_path_data[0]
    start_loc = gh.PLoc(long, lat)
    for next_ma in range(1, int(ma) + 1):
      # lat, long = ry.get_interpolated_yhs_position(yhs_path_data, next_ma)
      t1, lat, long = yhs_path_data[int(next_ma)]
      next_loc = gh.PLoc(long, lat)
      layer_paths.append(
          [(start_loc.long, start_loc.lat), (next_loc.long, next_loc.lat), f"rot step {ma}"])
      start_loc = next_loc
      self.add_run_paths_to_path_layer(layer_paths)
      # print(f"next_ma {next_ma}")
    return next_loc

  # plots path from start_point around compound translation and rotation Euler poles for specified time (ma)
  def RenderComboPoleMotionForMa(self, start_point, vPole, rPole, ma, N=20):
    layer_paths = []
    start_loc = start_point
    for i in range(0, N+1):
      next_ma = i * ma / N
      next_loc = ek.getCompoundRotationTranslationOfPoint(
          vPole, rPole, start_point, next_ma)
      layer_paths.append(
          [(start_loc.long, start_loc.lat), (next_loc.long, next_loc.lat), f"rot step {N}"])
      start_loc = next_loc
      self.add_run_paths_to_path_layer(layer_paths)
      # print (f"next_ma: {next_ma}")
    return next_loc

  # combo plot azimuth of cumilative rotation plot
  def RenderAzimuthMarkersforMa(self, start_point, vPole, rPole, ma, N=5):
    layer_paths = []
    start_loc = start_point
    for i in range(0, N+1):
      next_ma = (i + 1) * ma / N
      next_loc = ek.getCompoundRotationTranslationOfPoint(
          vPole, rPole, start_point, next_ma)
      rot_angle = -rPole.omega * i * ma / N
      mark_ploc = gh.getPointFromPavel(start_loc, gh.PAvel(
          rot_angle, .05), 1000)  # arbitrary length
      layer_paths.append(
          [(start_loc.long, start_loc.lat), (mark_ploc.long, mark_ploc.lat), f"mark step {N}"])
      start_loc = next_loc
      self.add_run_paths_to_path_layer(layer_paths)
    return next_loc

  # add arbitrary vector to layer
  def AddVectorToPath(self, start_loc, end_loc):
    layer_paths = [
        [(start_loc.long, start_loc.lat), (end_loc.long, end_loc.lat), f"Add vector "]]
    self.add_run_paths_to_path_layer(layer_paths)

  def clear_layer(self):
    # if self.path_layer :
    #   if self.path_layer.isValid():
    provider = self.qvector_layer.dataProvider()

    # 1. Gather all existing feature IDs in the layer
    all_ids = [f.id() for f in self.qvector_layer.getFeatures()]

    # 2. Tell the provider to delete them
    provider.deleteFeatures(all_ids)

    # 3. Force QGIS to redraw the layer (now completely empty)
    self.qvector_layer.triggerRepaint()

  def unload(self):
    layers_to_remove = QgsProject.instance().mapLayersByName(self.layer_name)
    for layer in layers_to_remove:
      QgsProject.instance().removeMapLayer(layer.id())
    self.qvector_layer = None
