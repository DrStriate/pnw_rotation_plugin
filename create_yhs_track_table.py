from qgis.core import (
    QgsVectorLayer,
    QgsField,
    QgsFeature,
    QgsProject,
    QgsGeometry
)
from qgis.PyQt.QtCore import QVariant

def createYHSTrackTable():

  # 1. Define the URI (e.g., "None?field=id:integer&field=name:string" for a non-spatial table)
  # Use "Point", "LineString", or "Polygon" instead of "None" if you need geometry.
  uri = "None?field=id:integer&field=name:string&field=score:double"

  # 2. Create the memory vector layer/table
  table_layer = QgsVectorLayer(uri, "YHs Track Table", "memory")

  # 3. Check if it's valid
  if table_layer.isValid():
      # 4. Add rows/features to the table
      dp = table_layer.dataProvider()
      
      feat1 = QgsFeature()
      feat1.setAttributes([1, "Alice", 95.5])
      
      feat2 = QgsFeature()
      feat2.setAttributes([2, "Bob", 88.0])
      
      dp.addFeatures([feat1, feat2])
      table_layer.updateExtents()

      # 5. Add the table/layer to the current QGIS project (so it shows up in the Layers panel)
      QgsProject.instance().addMapLayer(table_layer)
