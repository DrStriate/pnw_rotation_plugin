import gplately
# code from Gemini conversation AI/Gemini – Migrating pole model to Gplates 1

# Load your custom rotation file directly into GPlately
rotation_model = gplately.load_rotation_model("PNW_compound_model.rot")
# Set up the reconstruction engine
model = gplately.PlateReconstruction(rotation_model)
