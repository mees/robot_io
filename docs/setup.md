# Cameras

### Azure Kinect (Kinect 4)
- On Ubuntu 18 install azure kinect SDK with apt
- On Ubuntu 20 download libk4a*(-dev) and libk4abt*(-dev) from https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/
  and k4atools from https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools \
  Install with `sudo dpkg -i`

- Install Open3D in your Python env with `pip install open3d`

- For default usage, start `$ python robot_io/cams/kinect4/kinect4.py`

### Multiple Kinect Azure
- When multiple kinect azures are in use, we need to set the USB bandwidth to a higher value.
- Edit `/etc/default/grub`, replacing the line that says `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"` with `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=32"` for two Kinect Azure. Set the value `32` to higher `64` for three Kinect Azure.
- Run `sudo update-grub`
- Restart the computer

### RealSense SR300/SR305

First follow installation instructions for librealsense2 [here](https://github.com/IntelRealSense/librealsense)
```
pip install pyrealsense2
python robot_io/cams/realsense/realsenseSR300_librs2.py  # to test
```

### Framos D435e
```
groups | grep video  # user must be in video group, otherwise ask Michael K.
file /usr/src/librealsense2  # (see below a)
diff misc/framos_setup_files/setup.py /usr/src/librealsense2/setup.py
cp -r /usr/src/librealsense2 .
cd librealsense2
pip uninstall pyrealsense2
pip install -e .

cd ../robot_io/cams/realsense
python realsense.py  # test script
```
- a) If `/usr/src/librealsense2` does not exist, download FRAMOS software package from https://www.framos.com/en/industrial-depth-cameras#downloads. Follow installation instructions, make sure to use local admin user (e.g. xam2) to install (file system may NOT be network mounted). Alternatively, `wget http://hulc2.cs.uni-freiburg.de/downloads/librealsense2.zip`.
- b) Use Ethernet sockets on the ceiling for PoE.


# Robots

## Franka Emika Panda

### IK fast
IK fast is an analytic IK solver. In order to use IK fast, first install `ikfast-pybind`:
```
git clone --recursive https://github.com/yijiangh/ikfast_pybind
cd ikfast_pybind
# copy panda IK solution .cpp and .h to ikfast_pybind
cp ../robot_io/misc/ik_fast_files/ikfast.h ./src/franka_panda/
cp ../robot_io/misc/ik_fast_files/ikfast0x10000049.Transform6D.0_1_2_3_4_5_f6.cpp ./src/franka_panda/
pip install .
```
For creating different IK solutions (e.g. in case of a different gripper) please refer to:
`http://docs.ros.org/en/kinetic/api/framefab_irb6600_support/html/doc/ikfast_tutorial.html`

### Frankx
```
git clone git@github.com:lukashermann/frankx
cd frankx
git clone git@github.com:pantor/affx
git clone git@github.com:pantor/ruckig
cd affx; git checkout -b frankx_version dabe0ba; cd ..
cd ruckig; git checkout -b frankx_version 31f50f0; cd ..

conda install pybind11
vim setupy.py  # add "-DFranka_DIR=/opt/ros/noetic/share/franka/cmake/"
pip install -e .
firefox https://192.168.180.87/desk/  # unlock joints
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/:$LD_LIBRARY_PATH

cd robot_io/robot_interface
python panda_frankx_interface.py  # test robot
```

# Input Devices

## SpaceMouse
```
sudo apt install libspnav-dev spacenavd
conda activate robot
pip install spnav
```

Next test if it works, some common pitfalls are:
1. Turn on SpaceMouse in the back
2. May not work while charging.
3. Wireless range is quite limited.
4. Comment the following two lines in `site-packages/spnav/__init__.py`
```
#pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
#pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
```

To test execute the following program. When moving the mouse you should
see numbers scrolling by.
```
python robot_io/input_devices/space_mouse.py
```


## VR Teleoperation

### Install Steam and SteamVR
- In terminal run `$ steam`, it will start downloading an update and create a `.steam` folder in your home directory.
- If you get an error, try deleting the steam folders on your home directory with `rm -rf .local/share/Steam/` and `rm -rf .steam`
- In Steam, create user account or use existing account.
- Install SteamVR
  - If on `pickup` click `Steam -> Settings -> Downloads -> Steam Library Folders -> Add Library Folder -> /media/hdd/SteamLibrary` to add the existing installation of SteamVR to your Steam account
  - Otherwise download SteamVR
- Restart Steam
- Connect and turn on HTC VIVE
- Launch `Library -> SteamVR` (if not shown, check `[] Tools` box)
- If SteamVR throws an  `Error: setcap of vrcompositor-launcher failed`, run `$ sudo setcap CAP_SYS_NICE+ep /media/hdd/SteamLibrary/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher`
- Make sure Headset and controller are correctly detected
- Go through VR setup procedure (standing is sufficient)

### Install Bullet
```
$ git clone https://github.com/bulletphysics/bullet3.git
$ cd bullet3

# Optional: patch bullet for selecting correct rendering device
# (only relevant when using EGL and multi-gpu training)
$ wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow

# For building Bullet for VR  add -DUSE_OPENVR=ON to line 8 of build_cmake_pybullet_double.sh
# Run
$ ./build_cmake_pybullet_double.sh

$ pip install numpy  # important to have numpy installed before installing bullet
$ pip install -e .  # effectively this is building bullet a second time, but importing is easier when installing with pip

# add alias to your bashrc
alias bullet_vr="~/.steam/steam/ubuntu12_32/steam-runtime/run.sh </PATH/TO/BULLET/>bullet3/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory_VR"

# to test VR control
# make sure SteamVR is started
$ bullet_vr
$ cd <PATH/TO/ROBOTIO>/robot_io/robot_io/control
$ python teleop_robot.py
```

Robot Teleop instructions:
1. Push dead-man switch (riffled grip right)
2. Move controller in direction robot is pointing (twoards window)
3. Push top middle button (with three lines)
4. Robot should reset to home position
5. Robot only moves with dead-man-switch activated

### Marker Detector

```
$ cd robot_io/marker_detection/apriltag_detection
$ python setupBatch.py build_ext --inplace
```
