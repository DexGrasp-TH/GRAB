import os

folder = "/data/dataset/GRAB/extract/tools/object_meshes/contact_meshes"

object_names = sorted([os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".ply")])

print(object_names)
