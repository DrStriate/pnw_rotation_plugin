import json
import math
import gplately
import pygplates


def calculate_bearing(p1, p2):
  """Calculates the initial bearing (azimuth) from p1 to p2 on a sphere."""
  lat1, lon1 = map(math.radians, p1.to_lat_lon())
  lat2, lon2 = map(math.radians, p2.to_lat_lon())
  dlon = lon2 - lon1
  y = math.sin(dlon) * math.cos(lat2)
  x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
      math.cos(lat2) * math.cos(dlon)
  bearing = math.atan2(y, x)
  return (math.degrees(bearing) + 360) % 360


def calculate_haversine_distance(p1, p2):
  """Calculates great-circle distance between two points in radians without C++ typing bottlenecks."""
  lat1, lon1 = map(math.radians, p1.to_lat_lon())
  lat2, lon2 = map(math.radians, p2.to_lat_lon())

  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = math.sin(dlat/2)**2 + math.cos(lat1) * \
      math.cos(lat2) * math.sin(dlon/2)**2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  return c


def track_and_export_models():
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
  timeline_ma = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
  dt = 0.1  # 100,000-year step to compute near-instantaneous derivative vectors

  models = {
      "Muller2019": {"obj": rotation_model_2019, "name": "Müller et al. (2019)", "features": []},
      "Matthews2016": {"obj": rotation_model_2016, "name": "Matthews et al. (2016)", "features": []},
      "Seton2012": {"obj": rotation_model_2012, "name": "Seton et al. (2012)", "features": []}
  }

  print(f"========================================================================================================================")
  print(
      f" COMPREHENSIVE KINEMATIC & INSTANTANEOUS VELOCITY ANALYSIS FOR NORTH AMERICAN PLATE (ID {plate_id})")
  print(f"========================================================================================================================")

  for time_ma in timeline_ma:
    print(f"\nReconstruction Time: {time_ma} Ma")
    print("=" * 136)

    # Determine step boundaries for moving forward in time
    if time_ma == 0.0:
      t1, t2 = dt, 0.0  # At present day, look from 0.1 Ma heading into 0 Ma
    else:
      t1, t2 = time_ma, time_ma - dt  # Look from time t heading forward to t - dt

    for key, model_data in models.items():
      try:
        # A. FINITE ROTATION (Total displacement from 0 Ma to time_ma)
        finite_rot = model_data["obj"].get_rotation(
            time_ma, plate_id, anchor_plate_id)
        f_lat, f_lon, f_ang = finite_rot.get_lat_lon_euler_pole_and_angle_degrees()

        # Active tracking point for the volcanic hotspot track configuration
        reconstructed_point = finite_rot.get_inverse() * yhs_point
        y_lat, y_lon = reconstructed_point.to_lat_lon()

        # B. INSTANTANEOUS STAGE ROTATION DERIVATIVE
        rot1 = model_data["obj"].get_rotation(t1, plate_id, anchor_plate_id)
        rot2 = model_data["obj"].get_rotation(t2, plate_id, anchor_plate_id)
        stage_rot = rot2 * rot1.get_inverse()

        inst_lat, inst_lon, stage_angle = stage_rot.get_lat_lon_euler_pole_and_angle_degrees()
        omega = stage_angle / dt  # Angular velocity in deg/Myr

        # C. LOCAL POINT KINEMATICS (Calculate mm/yr and Azimuth heading forward in time)
        p1 = rot1.get_inverse() * yhs_point
        p2 = rot2.get_inverse() * yhs_point

        # FIXED: Call custom Python Haversine math function instead of buggy C++ signature link
        dist_rad = calculate_haversine_distance(p1, p2)
        # 1 km/Myr converts exactly to 1 mm/yr on Earth's crustal surface
        speed_mm_yr = (dist_rad * 6371.0) / dt
        azimuth = calculate_bearing(p1, p2)

        # Dynamic terminal print formatting
        print(f"  {model_data['name']:<23}")
        print(
            f"    └─ FINITE TOTAL : Pole Lat: {f_lat:>8.4f}° | Pole Lon: {f_lon:>9.4f}° | Net Angle: {f_ang:>8.4f}° | YHS: ({y_lat:.3f}°, {y_lon:.3f}°)")
        print(
            f"    └─ INSTANTANEOUS: Inst Lat: {inst_lat:>8.4f}° | Inst Lon: {inst_lon:>9.4f}° | Omega: {omega:>8.4f}°/Myr | Rate: {speed_mm_yr:>5.1f} mm/yr | Azimuth: {azimuth:>6.1f}°")
        print("-" * 136)

        # Store structural parameters for QGIS mapping
        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(y_lon), float(y_lat)]
            },
            "properties": {
                "Time_Ma": float(time_ma),
                "Model": model_data["name"],
                "Finite_Pole_Lat": float(f_lat),
                "Finite_Pole_Lon": float(f_lon),
                "Finite_Angle_Deg": float(f_ang),
                "Instant_Pole_Lat": float(inst_lat),
                "Instant_Pole_Lon": float(inst_lon),
                "Omega_Deg_Myr": float(omega),
                "Local_Speed_mm_yr": float(speed_mm_yr),
                "Local_Azimuth_Deg": float(azimuth)
            }
        }
        model_data["features"].append(geojson_feature)

      except Exception as e:
        print(
            f"  {model_data['name']:<23} -> Error calculating kinematics: {e}")
        print("-" * 136)

  # 3. Export to GeoJSON
  print(f"\n========================================================================================================================")
  print(" EXPORTING ENRICHED VECTOR LAYER DATASETS FOR QGIS")
  print(f"========================================================================================================================")
  for key, model_data in models.items():
    filename = f"yhs_kinematics_{key}.geojson"
    geojson_output = {
        "type": "FeatureCollection",
        "features": model_data["features"]
    }
    with open(filename, "w") as f:
      json.dump(geojson_output, f, indent=4)
    print(f" Successfully exported: {filename}")


if __name__ == "__main__":
  track_and_export_models()
