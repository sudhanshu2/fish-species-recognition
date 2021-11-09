#include <iostream>
#include <vector>
#include <string>

#include <chrono>

#include <atomic>
#include <exception>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "inference.h"
#include "background.h"
#include "utilities.h"
#include "parameters.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

atomic<bool> continue_computation;

mutex mutex_contour, mutex_buffer;
condition_variable transfer_contour, transfer_buffer;

bool receive_contour = false, receive_buffer = false;
bool send_contour = true, send_buffer = true;

vector< vector<Point> > shared_all_contour;
vector< Rect > shared_selected_bounding_boxes;

Mat shared_contour_frame;
Mat shared_buffer_frame;

int shared_batch_size;
int input_size;

void *shared_buffer_pointer;

void unlock_all_threads() {
  send_contour = true;
  send_buffer = true;
  receive_contour = true;
  receive_buffer = true;
  transfer_buffer.notify_all();
  transfer_contour.notify_all();
}

void stage_one() {
  while (continue_computation) {
    Mat frame = load_frame();

    vector< vector< Point > > contours;

    if (!frame.empty()) {
      contours = detect_foreground(frame);
    }

    unique_lock<mutex> lock_contour(mutex_contour);
    transfer_contour.wait(lock_contour,[](){return send_contour;});

    shared_contour_frame = frame;
    shared_all_contour = contours;

    receive_contour = true;
    send_contour = false;

    if (frame.empty()) {
      break;
    }

    lock_contour.unlock();
    transfer_contour.notify_all();
  }
}

void stage_two() {
  while (continue_computation) {
    vector< vector<Point> > contours;
    vector< Rect > bounding_boxes;
    Mat frame;

    unique_lock<mutex> lock_contour(mutex_contour);
    transfer_contour.wait(lock_contour, [](){return receive_contour;});

    contours = shared_all_contour;
    frame = shared_contour_frame;

    send_contour = true;
    receive_contour = false;

    lock_contour.unlock();
    transfer_contour.notify_all();

    int batch_size;

    if (frame.empty()) {
      batch_size = -1;
    } else {
      bounding_boxes = get_selected_contours(contours);
      batch_size = set_buffer(bounding_boxes, swap_buffer, frame);
    }

    unique_lock<mutex> lock_buffer(mutex_buffer);
    transfer_buffer.wait(lock_buffer, [](){return send_buffer;});

    if (continue_computation == false) {
      break;
    }

    cudaMemcpy(buffers[0], swap_buffer, input_size, cudaMemcpyDefault);

    shared_buffer_frame = frame;
    shared_selected_bounding_boxes = bounding_boxes;
    shared_batch_size = batch_size;

    receive_buffer = true;
    send_buffer = false;

    if (batch_size == -1) {
      break;
    }

    lock_buffer.unlock();
    transfer_buffer.notify_all();
  }
}

void stage_three() {
  while (continue_computation) {
    vector< Rect > bounding_boxes;
    Mat frame;
    int  batch_size;

    unique_lock<mutex> lock_buffer(mutex_buffer);
    transfer_buffer.wait(lock_buffer, [](){return receive_buffer;});

    bounding_boxes = shared_selected_bounding_boxes;
    frame = shared_contour_frame;
    batch_size = shared_batch_size;

    if (batch_size == -1) {
      break;
    }

    send_buffer = true;
    receive_buffer = false;

    lock_buffer.unlock();

    transfer_buffer.notify_all();

    if (batch_size > 0) {
      vector< string > inference_output = run_inference(batch_size, buffers[0]);
      if (SAVE_OUTPUT_TO_VIDEO) {
        output_frame(frame, inference_output, bounding_boxes, batch_size);
      }
    }

    cout << "\033[0m" << "current time " << (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count() << endl;

  }
}

int main() {
  continue_computation = true;

  input_size = initialize_species_recognition();
  initialize_background_modeling();
  initialize_video_capture();

  cout << "\033[0m" << "creating cpu buffer ... ";
  swap_buffer = malloc(input_size);
  if (swap_buffer == NULL) {
    cout << "\033[31m" << "failed for swap buffer with error " << endl;
    exit(1);
  }

  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "current time " << (duration_cast< milliseconds >(system_clock::now().time_since_epoch())).count() << endl;

  thread stage_one_thread(stage_one);
  thread stage_two_thread(stage_two);
  thread stage_three_thread(stage_three);

  stage_one_thread.join();
  stage_two_thread.join();
  stage_three_thread.join();
  video_destructor();
  return 0;
}
