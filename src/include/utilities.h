#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <vector>
#include <string>

#include "parameters.h"

#include <opencv2/opencv.hpp>

void initialize_video_capture();
cv::Mat load_frame();
void output_frame(cv::Mat, std::vector< std::string >, std::vector< cv::Rect >, int);
int set_buffer(std::vector< cv::Rect >, void*, cv::Mat);
std::vector< cv::Rect > get_selected_contours(std::vector< std::vector< cv::Point > >);

void video_destructor();

#endif
