#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include "Detection.h"
// #include "RunAprilDetector.hpp"
//
// int main(int argc, char **argv) {
//     cv::Mat image, image_gray;
// //     image = cv::imread("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample.png");
//     image = cv::imread("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample_kalibr.png");
//
//     namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//     imshow( "Display window", image );                   // Show our image inside it.
//
//
//     std::vector<Detection> det;
//     RunAprilDetector myDet("36h11b2");
//     det = myDet.processImage(image);
//
//     std::cout << "Det result: " << det.size() << " items \n";
//     for (int i=0; i < det.size(); ++i){
//         std::cout << "Type: " << det[i].type << "\n";
//         std::cout << "Id: " << det[i].id << "\n";
//         std::cout << "Point1: " << det[i].points[0].first << " / " << det[i].points[0].second << "\n";
//         std::cout << "\n";
//     }
//
//     return 0;
// }

#include <vector>
#include <string.h>

#include "RunAprilDetectorBatch.hpp"

int main(int argc, char **argv) {
    std::vector<std::string> imageBatch;
    imageBatch.push_back("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample_kalibr.png");
    imageBatch.push_back("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample_kalibr2.png");
    imageBatch.push_back("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample_kalibr.png");
    imageBatch.push_back("/home/zimmermc/projects/lmb_camcal/wrapped_apriltags/sample_kalibr2.png");

    RunAprilDetectorBatch myDetector("36h11b2", 2, 1.0, false);
    std::vector< std::vector<Detection> > det;
    det = myDetector.processImageBatch(imageBatch);
}
