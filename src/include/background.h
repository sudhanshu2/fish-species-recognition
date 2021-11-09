#ifndef BACKGROUND_H
#define BACKGROUND_H

#include <iostream>
#include <vector>
#include <string>

#include "parameters.h"

#include <opencv2/opencv.hpp>

void initialize_background_modeling();
std::vector< std::vector<cv::Point> > detect_foreground(cv::Mat);

#endif
