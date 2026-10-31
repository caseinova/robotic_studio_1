from pathlib import Path

import xacrodoc as xd

from robot_descriptions import ur3e_description


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


URDF_OUTPUT = (
    OUTPUT_DIR
    / "ur3e_official.urdf"
)

MJCF_OUTPUT = (
    OUTPUT_DIR
    / "ur3e_official.xml"
)

ASSET_DIR = (
    OUTPUT_DIR
    / "assets"
)

ASSET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Locate official Universal Robots description
# ============================================================

XACRO_PATH = Path(
    ur3e_description.XACRO_PATH
)

XACRO_ARGS = dict(
    ur3e_description.XACRO_ARGS
)


print()
print("==============================================")
print("Official Universal Robots UR3e source")
print("==============================================")

print()
print("Xacro:")
print(XACRO_PATH)

print()
print("Arguments:")
print(XACRO_ARGS)


# ============================================================
# Explicitly register the UR description package
#
# ur.urdf.xacro is inside:
#
#   <repo>/urdf/ur.urdf.xacro
#
# Therefore repo root is parent.parent
# ============================================================

UR_DESCRIPTION_ROOT = (
    XACRO_PATH.parent.parent
)

xd.packages.update_package_cache(
    {
        "ur_description":
            str(UR_DESCRIPTION_ROOT)
    }
)


print()
print("UR description package:")
print(UR_DESCRIPTION_ROOT)


# ============================================================
# Compile official Xacro
# ============================================================

print()
print("Compiling official UR3e Xacro...")

doc = xd.XacroDoc.from_file(
    str(XACRO_PATH),
    subargs=XACRO_ARGS,
)


# ============================================================
# Export flattened URDF
# ============================================================

print()
print("Writing flattened URDF...")

doc.to_urdf_file(
    str(URDF_OUTPUT),
    use_protocols=False,
)


print()
print("URDF written to:")
print(URDF_OUTPUT)


# ============================================================
# Copy mesh assets locally
#
# This gives our project its own local asset folder instead
# of depending on cached absolute paths.
# ============================================================

print()
print("Localising mesh assets...")

doc.localize_assets(
    str(ASSET_DIR)
)


# ============================================================
# Convert to MuJoCo MJCF
# ============================================================

print()
print("Converting URDF -> MJCF...")

doc.to_mjcf_file(
    str(MJCF_OUTPUT),

    # Asset filenames in generated MJCF will use
    # the local assets folder.
    strippath="true",
    meshdir="assets",

    # URDF visual meshes include DAE files.
    # MuJoCo conversion is most robust initially
    # using physical/collision geometry.
    discardvisual="true",
)


print()
print()
print("==============================================")
print("UR3e MuJoCo model generated successfully")
print("==============================================")

print()
print("MJCF:")
print(MJCF_OUTPUT)

print()
print("Assets:")
print(ASSET_DIR)