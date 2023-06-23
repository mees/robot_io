#ifndef DETECTION_H
#define DETECTION_H

#include <vector>
#include "string.h"

class Detection {
public:
    std::string type;
    int id;
    std::vector< std::pair<float, float> > points;

    Detection() { }

    Detection(std::string type, unsigned int id, std::vector< std::pair<float, float> > points):
        type(type),
        id(id),
        points(points)
    { }
};

#endif
