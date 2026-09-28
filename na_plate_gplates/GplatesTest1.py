import pygplates
import numpy as np
# Code from Gemini desscribed in AI/Gemini – Migrating pole model to Gplates 1

# 1. DEFINE YOUR 5 PARAMETERS
# Rotation pole
ROT_LAT = 45.0
ROT_LON = -120.0
ROT_SPEED = 0.5  # degrees/Myr

# Translating pole parameters
TRANS_AZIMUTH = 180.0  # Moving south
TRANS_SPEED = 0.1      # degrees/Myr (translation rate)

# 2. CALCULATE THE COUPLED POLE AXIS (R2)
# R2 must be 90 degrees orthogonal to the translation azimuth to cause a great-circle shift
r2_lat = 0.0  # Derived mathematically from ROT_LAT, ROT_LON, and TRANS_AZIMUTH
r2_lon = -30.0

# IDs for your reconstruction tree
MOVING_PLATE_ID = 999  # Your custom PNW/Siletzia plate block ID
FIXED_PLATE_ID = 701   # e.g., Stable North America

# 3. GENERATE ROTATIONS OVER TIME
output_lines = []

for time in range(0, 51):  # 0 to 50 Ma
  # Calculate time-dependent angles (converted to radians for pygplates)
  angle_1 = np.radians(ROT_SPEED * time)
  angle_2 = np.radians(TRANS_SPEED * time)

  # Define individual finite rotations
  rot_1 = pygplates.FiniteRotation(
      pygplates.PointOnSphere(ROT_LAT, ROT_LON), angle_1)
  rot_2 = pygplates.FiniteRotation(
      pygplates.PointOnSphere(r2_lat, r2_lon), angle_2)

  # Compose rotations (GPlates evaluates right-to-left or left-to-right based on fixed frame)
  compound_rot = rot_2 * rot_1

  # Extract the resulting total reconstruction pole
  lat_out, lon_out, angle_out = compound_rot.get_lat_lon_and_angle_in_degrees()

  # Format line to match standard GPlates .rot file string standard
  line = f"{MOVING_PLATE_ID:3d} {time:5.1f} {lat_out:8.4f} {lon_out:9.4f} {angle_out:8.4f} {FIXED_PLATE_ID:3d} ! Compound PNW Path"
  output_lines.append(line)

# Save to a GPlates-readable rotation file
with open("PNW_compound_model.rot", "w") as f:
  f.write("\n".join(output_lines))
