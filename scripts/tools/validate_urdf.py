#!/usr/bin/env python3
"""Validate the vendored TienKung URDF without requiring Isaac Sim."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = PROJECT_ROOT / "assets" / "robot_description" / "tienkung" / "tienkung2_lite.urdf"
EXPECTED_ACTUATED_JOINTS = 20


def main() -> int:
    root = ET.parse(URDF_PATH).getroot()
    joints = root.findall("joint")
    actuated = [joint for joint in joints if joint.attrib.get("type") != "fixed"]
    links = root.findall("link")
    missing_meshes = []

    for mesh in root.findall(".//mesh"):
        mesh_path = URDF_PATH.parent / mesh.attrib["filename"]
        if not mesh_path.is_file():
            missing_meshes.append(mesh_path)

    errors = []
    if len(actuated) != EXPECTED_ACTUATED_JOINTS:
        errors.append(f"expected {EXPECTED_ACTUATED_JOINTS} actuated joints, found {len(actuated)}")
    if missing_meshes:
        errors.append(f"missing {len(missing_meshes)} mesh files")
    for joint in actuated:
        if joint.find("limit") is None:
            errors.append(f"joint {joint.attrib['name']} has no limits")

    print(f"URDF: {URDF_PATH}")
    print(f"Links: {len(links)}")
    print(f"Actuated joints: {len(actuated)}")
    print(f"Referenced meshes: {len(root.findall('.//mesh'))}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

