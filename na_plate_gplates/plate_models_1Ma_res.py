import json
import gplately
import pygplates


def generate_high_res_tracks():
  # 1. Fetching the GPlately data models
  print("Fetching Müller et al. (2019) model...")
  server_2019 = gplately.DataServer("Muller2019")
  rotation_model_2019, _, _ = server_2019.get_plate_reconstruction_files()

  print("Fetching Matthews et al. (2016) model...")
  server_2016 = gplately.DataServer("Matthews2016")
  rotation_model_2016, _, _ = server_2016.get_plate_reconstruction_files()

  print("Fetching Seton et al. (2012) model...\n")
  server_2012 = gplately.DataServer("Seton2012")
  rotation_model_2012, _, _ = server_2012.get_plate_reconstruction_files()

  # 2. Setup your specific Yellowstone Hotspot (YHS) parameters
  yhs_lat = 44.43
  yhs_long = -110.67
  plate_id = 101       # North American Plate
  anchor_plate_id = 0  # Global spin axis / deep mantle frame

  yhs_point = pygplates.PointOnSphere(yhs_lat, yhs_long)

  # Build a continuous 1 Ma timeline dictionary map
  models = {
      "Muller2019": {"obj": rotation_model_2019, "name": "Müller et al. (2019)", "features": []},
      "Matthews2016": {"obj": rotation_model_2016, "name": "Matthews et al. (2016)", "features": []},
      "Seton2012": {"obj": rotation_model_2012, "name": "Seton et al. (2012)", "features": []}
  }

  print("==========================================================")
  print(" GENERATING HIGH-RESOLUTION CONTINUOUS 1 Ma TRACKS")
  print("==========================================================")

  # Generate 51 data points (from 0 Ma back to 50 Ma inclusive)
  for target_time in range(51):
    time_ma = float(target_time)

    for key, model_data in models.items():
      try:
        # Query finite rotation for Euler Pole
        finite_rot = model_data["obj"].get_rotation(
            time_ma, plate_id, anchor_plate_id)
        p_lat, p_lon, p_ang = finite_rot.get_lat_lon_euler_pole_and_angle_degrees()

        # Apply the precise lowercase get_inverse() method for volcanic hotspot tracking
        reconstructed_point = finite_rot.get_inverse() * yhs_point
        y_lat, y_lon = reconstructed_point.to_lat_lon()

        # Store structural parameters for QGIS mapping
        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(y_lon), float(y_lat)]
            },
            "properties": {
                "Time_Ma": time_ma,
                "Model": model_data["name"],
                "Euler_Lat": float(p_lat),
                "Euler_Lon": float(p_lon),
                "Euler_Ang": float(p_ang)
            }
        }
        model_data["features"].append(geojson_feature)

      except Exception as e:
        # Catch instances where models don't span the entire timeline duration
        pass

  # 3. Export to high-resolution GeoJSON files
  for key, model_data in models.items():
    filename = f"yhs_continuous_1ma_{key}.geojson"
    geojson_output = {
        "type": "FeatureCollection",
        "features": model_data["features"]
    }
    with open(filename, "w") as f:
      json.dump(geojson_output, f, indent=4)
    print(f" Successfully generated track file: {filename}")


if __name__ == "__main__":
  generate_high_res_tracks()
