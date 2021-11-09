#include "utilities.h"

using namespace std;
using namespace cv;

int frame_count = 0;

VideoCapture cap;
VideoWriter output;

const float TRANSFORM_STD_DEV [] = {
  TRANSFORM_STD_DEV_R * TRANSFORM_DIVISOR,
  TRANSFORM_STD_DEV_G * TRANSFORM_DIVISOR,
  TRANSFORM_STD_DEV_B * TRANSFORM_DIVISOR
};

const float TRANSFORM_MEAN [] = {
  TRANSFORM_MEAN_R * TRANSFORM_DIVISOR,
  TRANSFORM_MEAN_G * TRANSFORM_DIVISOR,
  TRANSFORM_MEAN_B * TRANSFORM_DIVISOR
};

void initialize_video_capture() {
  cout << "\033[0m" << "loading input video ... ";
  cap.open(VIDEO_INPUT_PATH);
  if (!cap.isOpened()) {
    cout << "\033[31m" << "error opening video input" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  const double CAPTURE_FPS = cap.get(CAP_PROP_FPS);
  const int FRAME_WIDTH = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  const int FRAME_HEIGHT = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

  cout << "\033[0m" << "setting up video output ... ";
  output.open(VIDEO_OUTPUT_PATH, CV_FOURCC('M','J','P','G'), CAPTURE_FPS, Size(FRAME_WIDTH, FRAME_HEIGHT));
  if (!output.isOpened()) {
    cout << "\033[31m" << "error setting up video output" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;
}

Mat load_frame() {
  Mat read_frame;

  if (frame_count > MAX_FRAMES) {
    return read_frame;
  }

  cap >> read_frame;

  if (read_frame.empty()) {
    cout << "empty frame" << endl;
    return read_frame;
  }

  frame_count++;
  
  return read_frame;
}

void output_frame(Mat frame, vector< string > inference_output, vector< Rect > bounding_boxes, int batch_size) {
  for (int i = 0; i < batch_size && bounding_boxes.size() > i; i++) {
    if (inference_output[i] == "fish") {
      rectangle(frame, bounding_boxes[i].tl(), bounding_boxes[i].br(), Scalar( 0, 255, 255 ), 12);
    }
  }
  output.write(frame);
}

int set_buffer(vector< Rect > bounding_boxes, void *cpu_buffer, Mat frame) {
  int batch_size = 0;
  float* databuffer = static_cast<float*>(cpu_buffer);

  for (int i = 0; i < bounding_boxes.size() && batch_size < MODEL_BATCH_SIZE; i++) {
    Mat cropped = frame(bounding_boxes[i]);
    resize(cropped, cropped, Size(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH));

    for (int pixel = 0; pixel < MODEL_INPUT_CHANNEL_SIZE; pixel++) {
      for (int channel = 0; channel < MODEL_INPUT_CHANNELS; channel++) {
        int column = pixel % MODEL_INPUT_HEIGHT;
        int row = pixel / MODEL_INPUT_WIDTH;

        /* convert NHWC to NCHW; convert bgr to rgb; normalize image to 0 - 1; subtract mean and standard deviation according to neural network specifications */
        databuffer[batch_size * MODEL_INPUT_IMAGE_SIZE + channel * MODEL_INPUT_CHANNEL_SIZE + pixel + 1] = (cropped.at<Vec3b>(row, column)[2 - channel] - TRANSFORM_MEAN[2 - channel]) / TRANSFORM_STD_DEV[2 - channel];
      }
    }

    batch_size++;
  }

  return batch_size;
}

vector< Rect > get_selected_contours(vector< vector<Point> > contours) {
  vector< Rect > selected_contour;
  int batch_size = 0;

  for (int i = 0; i < contours.size() && batch_size < MODEL_BATCH_SIZE; i++) {
    Rect countour_rect = boundingRect(contours[i]);
    if (countour_rect.height > MINIMUM_OBJECT_HEIGHT && countour_rect.width > MINIMUM_OBJECT_WIDTH) {
      selected_contour.push_back(countour_rect);
      batch_size++;
    }
  }

  return selected_contour;
}

void video_destructor() {
  cap.release();
  output.release();
}
