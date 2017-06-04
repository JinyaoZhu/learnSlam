#ifndef G2O_TYPE_H
#define G2O_TYPE_H

#include "myslam/common_include.h"



void bundleAdjustment (
    const vector< cv::Point3f > points_3d,
    const vector< cv::Point2f > points_2d,
    const cv::Mat& K,
    cv::Mat& R, cv::Mat& t );


#endif /* G2O_TYPE_H */