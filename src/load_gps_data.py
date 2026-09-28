from dataclasses import dataclass
import geopandas as gpd


@dataclass
class PState:
  longitude: float
  latitude: float
  vEast: float
  vNorth: float


class RotData:
  rotDataLayerName = 'nshm2023_GPS_velocity'
  XRange = {"min": -125.0, "max": -110.0}
  YRange = {"min": 37.0, "max": 50.0}
  XYSteps = 50
  XSize = (XRange["max"] - XRange["min"]) / XYSteps
  YSize = (YRange["max"] - YRange["min"]) / XYSteps

  # RotEntryExclusions = [] # no outliers taken out
  RotEntryExclusions = [438, 3659]  # clear outliers
  # RotEntryExclusions = [2425, 438, 3659] # more consistent final path

  def __init__(self):
    self.gdf = None

  def load(self):
    # Path to GPS rotation data
    zip_filename = 'data/NSHM2023_GPS_velocity.zip'
    self.gdf = gpd.read_file(f"zip://{zip_filename}")

    # Extract the Field List (Column Names and Data Types)
    # print("--- Field List (Columns) ---")
    # for column_name, data_type in gdf.dtypes.items():
    #     print(f"Field: {column_name:<15} | Type: {data_type}")

    # 4. Extract Data Vectors (Features)
    for idx, row in gpd.iterrows():
      print(
          f"Station {idx} lon: {row.lon:.4f}, lat: {row.lat:.4f}, Ve: {row.Ve:.4f}, Vn: {row.Vn: .4f}")
  return
