#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "TagDetection.h"
#include "TagFamily.h"
#include "FloatImage.h"

namespace AprilTags {

class TagDetector {
public:

	const TagFamily thisTagFamily;

	//! Constructor
        // note: TagFamily is instantiated here from TagCodes
        TagDetector(const TagCodes& tagCodes) : thisTagFamily(tagCodes) {}
        TagDetector(const TagCodes& tagCodes, int blackBorder) : thisTagFamily(tagCodes, blackBorder) {}

	std::vector<TagDetection> extractTags(const cv::Mat& image);

};

} // namespace

#endif
