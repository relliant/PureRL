"""Installation script for the PureRL Isaac Lab extension."""

from pathlib import Path

from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).parent

setup(
    name="purerl",
    version="0.1.0",
    description="Isaac Lab multi-terrain locomotion tasks for the TienKung humanoid",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10,<3.14",
    install_requires=["gymnasium"],
    zip_safe=False,
)

