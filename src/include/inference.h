#ifndef INFERENCE_H
#define INFERENCE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>


#include "parameters.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

extern void* buffers[2];
extern void* swap_buffer;

int initialize_species_recognition();
std::vector<std::string> run_inference(int, void*);

#endif
