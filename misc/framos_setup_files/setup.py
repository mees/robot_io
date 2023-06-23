import os
import pathlib
import sys

from setuptools import setup

cwd = pathlib.Path().absolute()

assert "librealsense2" in cwd.as_posix(), "this file should be copied to /usr/src/librealsense2, please check README"

# these dirs will be created in build_py, so if you don't have
# any python sources to bundle, the dirs will be missing
build_temp = pathlib.Path(os.path.join(cwd, "build"))
build_temp.mkdir(parents=True, exist_ok=True)

os.chdir(str(build_temp))
result = os.system(
    f"cmake {str(cwd)} -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE={sys.executable}"
)
print(result)
result = os.system("make -j$(nproc)")
print(result)
os.chdir(str(cwd))

setup(
    name="pyrealsense2",
    version="0.1",
    package_dir={"": "build/wrappers/python"},
)
