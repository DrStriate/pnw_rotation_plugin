from qgis._core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsMarkerSymbol,
    QgsSimpleMarkerSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsPalLayerSettings,
    QgsTextFormat,
    QgsVectorLayerSimpleLabeling,
    QgsField
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import QMetaType


class GeoWhiteboard:
  def __init__(self, base_vector_layer):
    """
    Initializes the whiteboard wrapper.
    :param base_vector_layer: Your primary QgsVectorLayer containing your arrows/velocities.
    """
    self.base_layer = base_vector_layer
    self.annotation_layer = None
    self._init_annotation_layer()
    # self.size = 4
    # self.stroke = 0.3

  def _init_annotation_layer(self):
    """Creates a hidden, temporary memory layer dedicated strictly to targets and labels."""
    crs_auth = self.base_layer.crs().authid()

    # 1. Create a memory point layer
    self.annotation_layer = QgsVectorLayer(
        f"Point?crs={crs_auth}", "Whiteboard Annotations", "memory")

    # 2. Add the attribute field using QgsField and QVariant
    self.annotation_layer.startEditing()
    self.annotation_layer.addAttribute(
        QgsField("label_text", QMetaType.Type(QMetaType.QString)))
    self.annotation_layer.commitChanges()

    # Configure the target symbology (Circle + Crosshairs)
    custom_symbol = QgsMarkerSymbol()
    strokeWidth = 0.4
    size = 4
    circle = QgsSimpleMarkerSymbolLayer()
    circle.setShape(QgsSimpleMarkerSymbolLayer.Circle)
    circle.setSize(size)
    circle.setColor(QColor("transparent"))
    circle.setStrokeColor(QColor("red"))
    circle.setStrokeWidth(strokeWidth)
    custom_symbol.changeSymbolLayer(0, circle)

    crosshair = QgsSimpleMarkerSymbolLayer()
    crosshair.setShape(QgsSimpleMarkerSymbolLayer.Cross)
    crosshair.setSize(size)
    crosshair.setStrokeColor(QColor("red"))
    crosshair.setStrokeWidth(strokeWidth)
    custom_symbol.appendSymbolLayer(crosshair)

    self.annotation_layer.setRenderer(QgsSingleSymbolRenderer(custom_symbol))

    # Configure the labeling engine to look at our 'label_text' field
    text_format = QgsTextFormat()
    text_format.setFont(QFont("Arial", 10))
    text_format.setColor(QColor("black"))

    buffer_settings = text_format.buffer()
    buffer_settings.setEnabled(True)
    buffer_settings.setSize(0.8)
    buffer_settings.setColor(QColor("white"))
    text_format.setBuffer(buffer_settings)

    label_settings = QgsPalLayerSettings()
    label_settings.setFormat(text_format)
    label_settings.fieldName = "label_text"
    label_settings.placement = QgsPalLayerSettings.OrderedPositionsAroundPoint
    label_settings.quadOffset = QgsPalLayerSettings.QuadrantRight
    label_settings.dist = 4.0

    self.annotation_layer.setLabeling(
        QgsVectorLayerSimpleLabeling(label_settings))
    self.annotation_layer.setLabelsEnabled(True)

    # Add the whiteboard annotations layer to the QGIS map registry
    # QgsProject.instance().addMapLayer(self.annotation_layer)
    # 1. Add the layer to the registry without automatically placing it in the legend
    QgsProject.instance().addMapLayer(self.annotation_layer, False)

    # 2. Explicitly insert it at the very top (index 0) of the layer tree
    root = QgsProject.instance().layerTreeRoot()
    root.insertLayer(0, self.annotation_layer)

  def draw_target(self, longitude, latitude, label=""):
    """Adds a target crosshair and an optional text label at a specified coordinate."""
    if not self.annotation_layer:
      self._init_annotation_layer()

    feat = QgsFeature(self.annotation_layer.fields())
    feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(longitude, latitude)))
    feat.setAttribute('label_text', label)

    self.annotation_layer.startEditing()
    self.annotation_layer.addFeature(feat)
    self.annotation_layer.commitChanges()
    self.annotation_layer.triggerRepaint()

  def clear_annotations(self):
    """Erases all ad-hoc targets and text labels completely, leaving your base vectors pristine."""
    if self.annotation_layer:
      self.annotation_layer.startEditing()
      # Delete every feature inside the whiteboard layer
      feats = [f.id() for f in self.annotation_layer.getFeatures()]
      self.annotation_layer.deleteFeatures(feats)
      self.annotation_layer.commitChanges()
      self.annotation_layer.triggerRepaint()

  def erase_everything(self):
    """Removes the whiteboard layer entirely from the project workspace."""
    if self.annotation_layer:
      QgsProject.instance().removeMapLayer(self.annotation_layer.id())
      self.annotation_layer = None

  def unload(self):
    """Removes the plugin menu items, actions, and cleans up workspace artifacts."""
    # Loop through and remove the actions built by Plugin Builder...
    # [Keep the default generated icon/menu removal logic here]

    # Explicitly target and destroy the whiteboard layer on exit
    self.erase_everything()
    layers_to_remove = QgsProject.instance().mapLayersByName("Whiteboard Annotations")
    for layer in layers_to_remove:
      QgsProject.instance().removeMapLayer(layer.id())
