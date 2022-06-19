# This example assumes we have a mesh object selected

target_pc = open(r"C:\Users\alpci\Desktop\Research Project\surface_registration\target_point_clouds\Optical\points1\sk1\pc4.txt")
lines = target_pc.readlines()

# line = lines[0].split(" ")

vertices = []

for line in lines:
    line_a = line.split(" ")
    vertices.append((float(line_a[0]), float(line_a[1]), float(line_a[2])))

print(lines)