#include "background.h"

using namespace std;
using namespace cv;

Mat kernel;
Scalar color;

Ptr<BackgroundSubtractorMOG2> model;

void initialize_background_modeling() {
  cout << "\033[0m" << "initializing background model ... ";
  kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
  color = Scalar(0, 0, 255);
  model = createBackgroundSubtractorMOG2();
  cout << "\033[32m" << "done" << endl;
}


vector< vector<Point> > detect_foreground(Mat frame) {
  Mat mask;
  vector< vector<Point> > contours;

  model->apply(frame, mask);

  dilate(mask, mask, kernel);
  findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  return contours;
}
