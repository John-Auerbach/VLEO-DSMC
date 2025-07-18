# tools/make_ampt_box.py

import os

filename = "surf/ampt_box.surf"
os.makedirs("surf", exist_ok=True)

points = [
    ( 0.5, -0.1, -0.1),  # 1
    ( 0.5, -0.1,  0.1),  # 2
    ( 0.5,  0.1,  0.1),  # 3
    ( 0.5,  0.1, -0.1),  # 4
    (-0.5, -0.1, -0.1),  # 5
    (-0.5, -0.1,  0.1),  # 6
    (-0.5,  0.1,  0.1),  # 7
    (-0.5,  0.1, -0.1),  # 8
]

triangles = [
    (1, 2, 3), (1, 3, 4),     # +x
    (5, 8, 7), (5, 7, 6),     # -x
    (4, 3, 7), (4, 7, 8),     # +y
    (1, 5, 6), (1, 6, 2),     # -y
    (2, 6, 7), (2, 7, 3),     # +z
    (1, 4, 8), (1, 8, 5),     # -z
]

with open(filename, "w") as f:
    f.write("# SPARTA surface file for centered 1×0.2×0.2 m box\n")
    f.write("Points\n8\n")
    for pt in points:
        f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    f.write("Triangles\n12\n")
    for tri in triangles:
        f.write(f"{tri[0]} {tri[1]} {tri[2]}\n")

print(f"Wrote surface to {filename}")