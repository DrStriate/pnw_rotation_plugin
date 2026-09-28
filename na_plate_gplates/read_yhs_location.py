import json
import os
import math


def load_data(geojson_path: str):
  """ 
  Reads a pre-computed kinematics/track GeoJSON file and returns a data structuure
  Parameters:
      geojson_path (str): Full system file path to the output track file.
  """
  if not os.path.exists(geojson_path):
    raise FileNotFoundError(f"Target track layer not found at: {geojson_path}")

  with open(geojson_path, 'r') as f:
    data = json.load(f)

  features = data.get("features", [])
  if not features:
    return None, None

  # Extract structural coordinate maps and filter by available age parameters
  datapoints = []
  for feat in features:
    props = feat.get("properties", {})
    geom = feat.get("geometry", {})

    if "Time_Ma" in props and geom.get("type") == "Point":
      time = float(props["Time_Ma"])
      lon, lat = geom["coordinates"]  # GeoJSON standard structure pairs [X, Y]
      datapoints.append((time, lat, lon))

  # Sort data chronologically to ensure binary searching works smoothly
  datapoints.sort(key=lambda x: x[0])
  return datapoints


def get_interpolated_yhs_position(datapoints, age_ma: float) -> tuple:
  """
  returns the estimated (Latitude, Longitude) for any requested target age (Ma).

  Bypasses spatial dependencies entirely for fast, online QGIS plugin processing.

  Parameters:
      datapoints: structure derived from file read method above
      age_ma (float): The desired target reconstruction age in millions of years.

  Returns:
      tuple: (Latitude, Longitude) as decimal floats, or (None, None) if out of bounds.
  """
  # Boundary constraints safeguard checks
  if age_ma <= datapoints[0][0]:
    return datapoints[0][1], datapoints[0][2]
  if age_ma >= datapoints[-1][0]:
    return datapoints[-1][1], datapoints[-1][2]

  # Search for matching bounding blocks to handle precise interval placement
  for i in range(len(datapoints) - 1):
    t1, lat1, lon1 = datapoints[i]
    t2, lat2, lon2 = datapoints[i+1]

    # Exact integer match found
    if math.isclose(age_ma, t1):
      return lat1, lon1

    # Target age falls squarely between these two track increments
    if t1 <= age_ma <= t2:
      # Calculate interpolation weight percentage factor
      fraction = (age_ma - t1) / (t2 - t1)

      # Interpolate coordinates linearly for narrow time frames
      interp_lat = lat1 + fraction * (lat2 - lat1)
      interp_lon = lon1 + fraction * (lon2 - lon1)
      return interp_lat, interp_lon

  return None, None
