
#include "types/Camera.h"

namespace RGBDSLAM
{

    Camera::Camera(double fx, double fy, double cx, double cy) : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    Mat33 Camera::K()
    {
        Mat33 k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }
}
