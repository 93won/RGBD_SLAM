
#include "types/Feature.h"

namespace RGBDSLAM
{

    Feature::Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth) : frame_(frame),
                                                                                           position_(kp),
                                                                                           depth_(depth) {}

    Feature::Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth, std::vector<double> rgb) : frame_(frame),
                                                                                                                    position_(kp),
                                                                                                                    depth_(depth),
                                                                                                                    rgb_(rgb) {}

}
