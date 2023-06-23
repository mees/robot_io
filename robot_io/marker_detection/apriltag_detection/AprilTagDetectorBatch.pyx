# distutils: language = c++

from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np  # for np.ndarray
from cpython.string cimport PyString_AsString
from libc.stdlib cimport free, malloc
from libc.string cimport strcmp


# some OpenCV matrix types
cdef extern from "opencv2/opencv.hpp":
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_64FC1

cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int, void*) except +
        void create(int, int, int)
        void* data
        int type() const
        int cols
        int rows
        int channels()
        Mat clone() const

cdef void array2mat(np.ndarray arr, Mat& mat):
    cdef int r = arr.shape[0]
    cdef int c = arr.shape[1]
    cdef int mat_type = CV_8UC3            # or CV_64FC1, or CV_8UC3, or whatever
    mat.create(r, c, mat_type)
    cdef unsigned int px_size = 3           # 8 for single-channel double image or
                                            #   1*3 for three-channel uint8 image
    memcpy(mat.data, arr.data, r*c*px_size)

cdef vector[string] to_cstring_array(list_str):
    cdef vector[string] ret
    cdef string tmp
    for i in xrange(len(list_str)):
        if isinstance(list_str[i], unicode):
            ret.push_back(list_str[i].encode('UTF-8'))
        else:
            ret.push_back(list_str[i])

    return ret

##########
""" DETECTION DATA CONTAINER"""

# Makes the cpp class available in here
cdef extern from "Detection.h":
    cdef cppclass Detection:
        Detection() except +
        Detection(string, int, vector[pair[float, float]]) except +
        string type
        int id
        vector[pair[float, float]] points

cdef class PyDetection:
    cdef Detection* c_Detection      # holds pointer to an C++ instance which we're wrapping

    def __cinit__(self, type, int id, vector[pair[float, float]] points):

        # type is byte or unicode, but std::string wants bytes
        if isinstance(type, unicode):
            type = type.encode('UTF-8')

        self.c_Detection = new Detection(type, id, points)

    def __dealloc__(self):
        del self.c_Detection

    @property
    def type(self):
        return self.c_Detection.type.decode('UTF-8')

    @type.setter
    def type(self, value):
        if isinstance(value, unicode):
            value = value.encode('UTF-8')
        self.c_Detection.type = value

    @property
    def id(self):
        return self.c_Detection.id

    @id.setter
    def id(self, value):
        self.c_Detection.id = value

    @property
    def points(self):
        return self.c_Detection.points

    @points.setter
    def points(self, value):
        self.c_Detection.points = value

# Factory for creating the python equivalent of the c class
cdef object PyDetection_factory(Detection tmp):
    cdef string type_str

    if isinstance(tmp.type, unicode):
        type_str = tmp.type.encode('UTF-8')
    else:
        type_str = tmp.type
    cdef PyDetection py_obj = PyDetection(type_str, tmp.id, tmp.points)
    return py_obj


###########
#""" APRILTAG DETECTOR BATCH"""

cdef extern from "RunAprilDetectorBatch.hpp":
    # Declares that we want to use this class here
    cdef cppclass RunAprilDetectorBatch:
        # list attributes and methods we are going to use
        RunAprilDetectorBatch(string) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        RunAprilDetectorBatch(string, bool) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        RunAprilDetectorBatch(string, int, unsigned int, bool, float) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        vector[vector[Detection]] processImageBatch(vector[string])
        vector[Detection] processImage(string)
        vector[Detection] processImageM(Mat& image)


cdef class PyRunAprilDetectorBatch:
    cdef RunAprilDetectorBatch* c_RunAprilDetectorBatch      # holds pointer to an C++ instance which we're wrapping
    def __cinit__(self, codeName, int blackBorder, unsigned int maxNumThreads, float resizeFactor, bool draw=False):

        if isinstance(codeName, unicode):
            codeName = codeName.encode('UTF-8')
        self.c_RunAprilDetectorBatch = new RunAprilDetectorBatch(codeName, blackBorder, maxNumThreads, draw, resizeFactor)

    def __dealloc__(self):
        del self.c_RunAprilDetectorBatch

    def processImageBatch(self, imagePaths):
        cdef vector[string] imagePathsEnc = to_cstring_array(imagePaths)
        # imagePathsEnc = list()
        # for x in imagePaths:
        #    imagePathsEnc.append(x.encode('UTF-8'))

        # cdef vector[string] imagePathsEnc
        # for x in imagePaths:
        #     imagePathsEnc.push_back(x.encode('UTF-8'))

        # get the c class result
        cdef vector[vector[Detection]] cResult = self.c_RunAprilDetectorBatch.processImageBatch(imagePathsEnc)

        fullOut = list()
        for imgResult in cResult:
            imgOut = list()
            for x in imgResult:
                imgOut.append(PyDetection_factory(x))
            fullOut.append(imgOut)

        return fullOut

    def processImage(self, imagePath):
        cdef string tmp
        if isinstance(imagePath, unicode):
            tmp = imagePath.encode('UTF-8')
        else:
            tmp = imagePath

        cdef vector[Detection] imgResult = self.c_RunAprilDetectorBatch.processImage(tmp)
        imgOut = list()
        for x in imgResult:
            imgOut.append(PyDetection_factory(x))
        return imgOut

    def processImageM(self, np.ndarray[np.uint8_t, ndim=3] image):
        cdef Mat image_mat
        array2mat(image, image_mat)
        cdef vector[Detection] imgResult = self.c_RunAprilDetectorBatch.processImageM(image_mat)
        imgOut = list()
        for x in imgResult:
            imgOut.append(PyDetection_factory(x))
        return imgOut
