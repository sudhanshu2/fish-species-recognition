# Fish Species Recognition :fish: 

This repository contains the code for the manuscript "Is it Nemo or Dory? Fast and accurate object detection for IoT and edge devices" in 11th International Conference on the Internet of Things. Please cite our project if you use this code.

## Getting Started

### Installing Dependencies

This project has been tested on Jetson Nano and it uses its unified memory architecture to implement CPU-GPU split in computation. In order to replicate the software stack that was used in the test device, follow the instructions below after setting-up the device to use the headless mode using instructions [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#setup-headless).


Next step is installing a few dependencies, and uninstalling the pre-installed OpenCV library.
```
sudo apt update
sudo apt upgrade
sudo apt remove libopencv libopencv-dev libopencv-python libopencv-samples opencv-licenses vpi1-samples
sudo apt-get install libhdf5-serial-dev libopencv-viz-dev libfreetype6-dev libv4l-dev libavcodec-dev libavformat-dev libgtk-3-dev gtk2.0 libswscale-dev libavresample-dev libtesseract-dev libdc1394-utils libopenblas-dev liblapacke-dev libgoogle-glog-dev libvtk6-dev libgflags-dev libogre-1.9-dev libboost-all-dev
```

Then download OpenCV, and its contrib library. The specific version used was 3.4.13 due to its compatibility with most of the algorithms in the BGS library unlike OpenCV 4.

```
git clone --depth 1 --branch 3.4.13 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 3.4.13 https://github.com/opencv/opencv_contrib.git
```

Now, build and make OpenCV.

```
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D WITH_CUDA=ON -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D WITH_CUBLAS=ON -D WITH_QT=OFF -D OPENCV_GENERATE_PKGCONFIG=ON -D WITH_CUDNN=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D WITH_FFMPEG=ON -D BUILD_EXAMPLES=OFF -D WITH_LIBV4L=ON -D WITH_GTK=ON ..
make -j4
sudo make install
```

Add path to a trained model and the input video stream in the `parameters.h` file. Also add all potential labels to `inference.cpp`. To run the code,
```
git clone https://github.com/sudhanshu2/fish-recognizer.git
cd fish-recognizer
mkdir build
cd build
cmake ..
make -j4
./fishspecies
```

## Dataset
The dataset for this project used [fishfish](https://kaggle.com/tomeryacov/fishfish) dataset on Kaggle, the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and photos captured by the authors at the Georgia Aqaurium.

## Citation
```
@inproceedings{agarwal2021nemo,
  title={Is it Nemo or Dory? Fast and accurate object detection for IoT and edge devices},
  author={Agarwal, Sudhanshu and Vuduc, Richard},
  booktitle={11th International Conference on the Internet of Things},
  pages={94--101},
  year={2021}
}
```
---

This project was developed at the HPC Garage research group at the Georgia Institute of Technology.
