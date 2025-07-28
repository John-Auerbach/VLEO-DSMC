#!/bin/bin/env python

# Script:  stl2surf.py
# Purpose: convert an STL file (binary) into a SPARTA surface file
#          and warn if surface is not watertight
# Author:  Steve Plimpton (Sandia), sjplimp at sandia.gov (modified by John Auerbach, Penn State)
# Syntax:  stl2surf.py stlfile surffile
#          stlfile = read this stereolithography (STL) file
#                    in binary format (ASCII also accepted)
#          surffile = write this SPARTA surface file

# NOTE: process vertices in text format so no precision or round-off issues
# NOTE: if the STL is binary, convert it to ASCII text in memory first

from __future__ import print_function

# error message

def error(str=None):
  if str: print("ERROR:",str)
  else: print("Syntax: stl2surf.py stlfile surffile")
  sys.exit()

# ----------------------------------------------------------------------
# main program

import sys,re,struct

if len(sys.argv) != 3: error()

stlfile = sys.argv[1]
surffile = sys.argv[2]

# parse STL file into triangles and triangle vertices
# tritxt = list of text between facet and endfacet
# triverts = list of 3 vertices per triangle, in text format

def load_stl_as_text(path):
  # try ASCII first
  with open(path, "rb") as f:
    data = f.read()
  try:
    txt = data.decode("utf-8")
    if txt.lstrip().startswith("solid") and re.search(r"\bfacet\b", txt) and re.search(r"\bvertex\b", txt):
      return txt
  except UnicodeDecodeError:
    pass
  # binary → synthesize ASCII text
  if len(data) < 84:
    error("STL file %s has incorrect format" % path)
  header = data[:80]
  try:
    name = header.decode("ascii", errors="ignore").strip().splitlines()[0]
  except Exception:
    name = ""
  ntri = struct.unpack("<I", data[80:84])[0]
  offset = 84
  out = [f"solid {name}"]
  need = 84 + ntri*50
  if len(data) < need:
    error("STL file %s truncated (binary STL)" % path)
  for _ in range(ntri):
    nx, ny, nz, x1, y1, z1, x2, y2, z2, x3, y3, z3, abc = struct.unpack("<12fH", data[offset:offset+50])
    offset += 50
    out.append(f"  facet normal {nx} {ny} {nz}")
    out.append("    outer loop")
    out.append(f"      vertex {x1} {y1} {z1}")
    out.append(f"      vertex {x2} {y2} {z2}")
    out.append(f"      vertex {x3} {y3} {z3}")
    out.append("    endloop")
    out.append("  endfacet")
  out.append(f"endsolid {name}")
  return "\n".join(out) + "\n"

stltxt = load_stl_as_text(stlfile)

match = re.search("^.*\n",stltxt)
if not match: error("STL file %s has incorrect format" % stlfile)
words = match.group().split()
if words[0] != "solid":
  error("STL file %s has incorrect format" % stlfile)
if len(words) >= 2: name = words[1]
else: name = ""

tritxt = re.split("endfacet\s+facet",stltxt)
print("# of triangles in STL file:",len(tritxt))

pattern = ".*vertex\s+(\S+)\s+(\S+)\s+(\S+)"
triverts = []

for one in tritxt:
  vertices = re.findall(pattern,one)
  if len(vertices) != 3:
    print("Triangle record:")
    print(one)
    error("Invalid triangle vertices")
  triverts.append(vertices)

# build list of unique vertices via hash
# unique: key = vertex 3-tuple, value = index in verts
# verts = list of unique vertices, each as 3-tuple
# tris = list of 3 vertex indices for each triangle

unique = {}
verts = []
tris = []

for vert3 in triverts:
  if vert3[0] in unique: v0 = unique[vert3[0]]
  else:
    v0 = len(verts)
    verts.append(vert3[0])
    unique[vert3[0]] = v0
  if vert3[1] in unique: v1 = unique[vert3[1]]
  else:
    v1 = len(verts)
    verts.append(vert3[1])
    unique[vert3[1]] = v1
  if vert3[2] in unique: v2 = unique[vert3[2]]
  else:
    v2 = len(verts)
    verts.append(vert3[2])
    unique[vert3[2]] = v2
  tris.append((v0,v1,v2))

# print SPARTA surface file

fp = open(surffile,"w")

fp.write("#surf file\n\n")
fp.write("%d points\n" % len(verts))
fp.write("%d triangles\n\n" % len(tris))

fp.write("Points\n\n")
for i,vert in enumerate(verts):
  fp.write("%d %s %s %s\n" % (i+1, vert[0], vert[1], vert[2]))

fp.write("\nTriangles\n\n")
for i,tri in enumerate(tris):
  fp.write("%d %d %d %d\n" % (i+1, tri[0]+1, tri[1]+1, tri[2]+1))

fp.close()
  
# stats to screen

print("# of vertices in SPARTA file:",len(verts))
print("# of triangles in SPARTA file:",len(tris))

# warn if not a watertight object
# watertight = each edge is part of exactly 2 triangles
#   and edge direction is different in 2 triangles

ehash = {}
dup = 0
unmatch = 0

for vert3 in triverts:
  edge = (vert3[0],vert3[1])
  if edge in ehash:
    dup += 1
    dupedge = edge
  else: ehash[edge] = 1
  edge = (vert3[1],vert3[2])
  if edge in ehash:
    dup += 1
    dupedge = edge
  else: ehash[edge] = 1
  edge = (vert3[2],vert3[0])
  if edge in ehash:
    dup += 1
    dupedge = edge
  else: ehash[edge] = 1

for edge in ehash:
  invedge = (edge[1],edge[0])
  if invedge not in ehash:
    unmatch += 1
    unmatchedge = edge
    
if dup or unmatch:
  print("WARNING: surface is not watertight")

if dup:
  print("Duplicate edge count:",dup)
  print("One duplicate edge:",dupedge)

if unmatch:
  print("Unmatched edge count:",unmatch)
  print("One unmatched edge:",unmatchedge)
