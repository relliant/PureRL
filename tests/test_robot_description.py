import xml.etree.ElementTree as ET

from purerl.config.robot import DEFAULT_URDF_PATH

URDF = DEFAULT_URDF_PATH


def test_urdf_has_expected_joint_count_and_limits():
    robot = ET.parse(URDF).getroot()
    actuated = [joint for joint in robot.findall("joint") if joint.attrib["type"] != "fixed"]

    assert len(actuated) == 20
    assert all(joint.find("limit") is not None for joint in actuated)


def test_all_referenced_meshes_exist():
    robot = ET.parse(URDF).getroot()
    missing = [
        mesh.attrib["filename"]
        for mesh in robot.findall(".//mesh")
        if not (URDF.parent / mesh.attrib["filename"]).is_file()
    ]

    assert missing == []


def test_expected_contact_and_root_links_exist():
    robot = ET.parse(URDF).getroot()
    links = {link.attrib["name"] for link in robot.findall("link")}

    assert {"pelvis", "ankle_roll_l_link", "ankle_roll_r_link"} <= links
