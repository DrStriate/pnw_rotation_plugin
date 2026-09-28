import os
import read_yhs_location as ry

if __name__ == "__main__":

  # 1. Dynamically compute the path to the directory containing this script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # 2. Join the script directory with the filename to create a local absolute path
  file_path = os.path.join(script_dir, "yhs_continuous_1ma_Muller2019.geojson")

  # Quick test case query parameters (e.g., 15.4 Ma)
  test_age = 15.4

  print(f"--- QGIS Plugin Function Verification ---")
  print(f"Searching locally at: {file_path}")

  try:
    datapoints = ry.load_data(file_path)
    lat, lon = ry.get_interpolated_yhs_position(datapoints, test_age)
    if lat is not None:
      print(f"Target Age: {test_age} Ma")
      print(
          f"Estimated Yellowstone Location -> Lat: {lat:.4f}° | Lon: {lon:.4f}°")
    else:
      print(
          f"Error: Age {test_age} Ma falls outside the data boundaries of the GeoJSON file.")
  except FileNotFoundError as fnf:
    print(f"Error: {fnf}")
    print("Please verify the master script has run and saved the files in this directory.")
  except Exception as e:
    print(f"Verification tracking execution failed: {e}")
