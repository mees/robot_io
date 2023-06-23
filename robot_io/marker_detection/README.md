# Installation
Ideally build OpenCV:

    https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/

Dependency for the TagDetector:

    sudo apt install libeigen3-dev

Python packages needed:

    opencv-python tqdm numpy imageio pyx Pillow matplotlib scipy commentjson

Build the TagDetector:

    cd apriltag_detection && python setupBatch.py build_ext --inplace

# Credits

For AprilTag Detection and Marker Creation:

    https://github.com/zimmerm/FreiCalib
    Which is based on Kaess Apriltag C++ Detector: https://people.csail.mit.edu/kaess/apriltags/
