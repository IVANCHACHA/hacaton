import os
from glob import glob

from setuptools import setup, find_packages

package_name = "sorter_pak"


def data_files_for_dir(src_dir, share_subdir):
    """Recursively install every file under `src_dir` (e.g. models/, with
    its models/<slug>/meshes/*.stl subfolders) preserving structure under
    share/<package>/<share_subdir>/... - glob() alone only handles one
    level, and the Gazebo models have a nested meshes/ folder each."""
    entries = []
    for root, _dirs, files in os.walk(src_dir):
        if not files:
            continue
        rel = os.path.relpath(root, src_dir)
        dest = os.path.join("share", package_name, share_subdir) if rel == "." else \
            os.path.join("share", package_name, share_subdir, rel)
        entries.append((dest, [os.path.join(root, f) for f in files]))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        *data_files_for_dir("worlds", "worlds"),
        *data_files_for_dir("models", "models"),
    ],
    install_requires=["setuptools", "numpy", "scipy", "opencv-python"],
    zip_safe=True,
    maintainer="sorter_pak team",
    maintainer_email="dev@example.com",
    description="Track 3 PAK: B/C/D classification + 2-shiber executive routing.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "conveyor_sim_node = sorter_pak.conveyor_sim_node:main",
            "perception_node = sorter_pak.perception_node:main",
            "executive_node = sorter_pak.executive_node:main",
            "gz_bridge_node = sorter_pak.gz_bridge_node:main",
            "build_catalog = sorter_pak.core.stl_dataset:main_cli",
        ],
    },
)
