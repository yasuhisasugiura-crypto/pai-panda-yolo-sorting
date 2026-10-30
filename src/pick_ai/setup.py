from pathlib import Path
from setuptools import find_packages, setup

base = Path(__file__).resolve().parent
for required in ("setup.cfg", "resource/pick_ai", "pick_ai/__init__.py",
                 "pick_ai/geometry.py", "pick_ai/perception_node.py",
                 "pick_ai/object_spawner.py"):
    if not (base / required).is_file():
        raise RuntimeError(f"Missing ROS Python package file: {required}; use the complete package")

package_name = "pick_ai"
setup(
    name=package_name,
    version="0.4.4",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="PAI student",
    maintainer_email="noreply@example.invalid",
    description="Perception + random spawner for the Panda sorting demo (Stage A/B/C)",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "perception_node = pick_ai.perception_node:main",
            "object_spawner = pick_ai.object_spawner:main",
        ],
    },
)

