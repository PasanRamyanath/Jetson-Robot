#!/usr/bin/env python3
"""Make slam_toolbox (humble) build headless (§7.3): drop the rviz plugin, Qt5 and the unused G2O find.

The core nodes only need interactive_markers (loop-closure assistant); rviz/Ogre/Qt would add ~1 h of build and
~150 MB of image for a plugin that never runs on the robot. Idempotent; fails loudly if upstream changed shape.
Usage: patch_slam_toolbox.py /ws/src/slam_toolbox
"""
import re
import sys

root = sys.argv[1]
path = root + "/CMakeLists.txt"
src = open(path).read()
orig = src

DROP_PKGS = ("rviz_common", "rviz_default_plugins", "rviz_ogre_vendor", "rviz_rendering", "Qt5", "G2O")
for p in DROP_PKGS:
    src = re.sub(r"^\s*find_package\(%s\b[^)]*\)\s*\n" % p, "", src, flags=re.M)
    src = re.sub(r"^\s*%s\s*\n" % p, "", src, flags=re.M)                     # entries in set(dependencies ...)
src = re.sub(r"#### rviz Plugin.*?(?=#### Ceres Plugin)", "", src, flags=re.S)
src = re.sub(r"^\s*SlamToolboxPlugin\s*\n", "", src, flags=re.M)              # set(libraries ...)
src = re.sub(r"install\(TARGETS SlamToolboxPlugin.*?\)\s*\n", "", src, flags=re.S)
src = re.sub(r"^ament_export_targets\(SlamToolboxPlugin[^)]*\)\s*\n", "", src, flags=re.M)
src = src.replace("solver_plugins.xml rviz_plugins.xml", "solver_plugins.xml")

if "rviz" in src.replace("rviz_plugins.xml", "") or "SlamToolboxPlugin" in src or "Qt5" in src:
    if src != orig:
        open(path + ".rej", "w").write(src)
    sys.exit("patch_slam_toolbox: upstream CMakeLists changed; see %s.rej" % path)
open(path, "w").write(src)

pkg = root + "/package.xml"
x = open(pkg).read()
x = re.sub(r"^\s*<(?:build_|exec_)?depend>(?:rviz_\w+|libqt5-\w+|qtbase5-dev)</(?:build_|exec_)?depend>\s*\n", "",
           x, flags=re.M)
x = re.sub(r"^\s*<rviz_common [^>]*/>\s*\n", "", x, flags=re.M)
open(pkg, "w").write(x)

# Patch karto_sdk to find TBB on Ubuntu 18.04 (no TBBConfig.cmake in 18.04's libtbb-dev)
import os, shutil
karto_path = os.path.join(root, "lib", "karto_sdk", "CMakeLists.txt")
if os.path.exists(karto_path):
    k_src = open(karto_path).read()
    tbb_block = """
find_package(TBB QUIET)
if(NOT TARGET TBB::tbb)
  find_library(TBB_LIB NAMES tbb)
  find_path(TBB_INC NAMES tbb/tbb.h)
  add_library(TBB::tbb UNKNOWN IMPORTED)
  set_target_properties(TBB::tbb PROPERTIES
    IMPORTED_LOCATION "${TBB_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INC}"
  )
endif()
"""
    k_src = re.sub(r"find_package\(TBB\b[^)]*\)", tbb_block.strip(), k_src)
    open(karto_path, "w").write(k_src)
    print("karto_sdk TBB discovery patched")

cmake_dir = os.path.join(root, "CMake")
lower_cmake_dir = os.path.join(root, "cmake")
if os.path.isdir(cmake_dir) and not os.path.exists(lower_cmake_dir):
    try:
        shutil.copytree(cmake_dir, lower_cmake_dir)
    except Exception:
        pass

print("slam_toolbox patched headless")
