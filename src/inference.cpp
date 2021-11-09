#include "inference.h"

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

IBuilder *builder;
INetworkDefinition* network;
IBuilderConfig* config;
IParser* parser;
ICudaEngine* engine;
IExecutionContext* context;

void* buffers[2];
void* swap_buffer;

/* Add the potential labels for classification to the array below */
const string MODEL_CLASS_LABELS[1] = {"none"}; 

int inputSize = 1;
int outputSize = 1;
nvinfer1::Dims inputDims, outputDims;

class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) || (severity == Severity::kINFO && SHOW_INFO)) {
      cout << msg << endl;
    }
  }
 } gLogger;

struct InferDeleter {
template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

int initialize_species_recognition() {
  cout << "\033[0m" << "initializing builder ... ";
  builder = createInferBuilder(gLogger);
  builder->setMaxBatchSize(MODEL_BATCH_SIZE);
  if (!builder) {
    cout << "\033[31m" << "failed" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "initializing network definition ... ";
  NetworkDefinitionCreationFlags explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network = builder->createNetworkV2(explicitBatch);
  if (!network) {
    cout << "\033[31m" << "failed" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "initializing builder config ... ";
  config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1ULL << 30);
  if (builder->platformHasFastFp16() && ENABLE_FP16) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (!config) {
    cout << "\033[31m" << "failed" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "initializing parser ... " << endl;
  parser = createParser(*network, gLogger);
  parser->parseFromFile(MODEL_PATH, static_cast<int>(ILogger::Severity::kINFO));
  if (!parser) {
    cout << "\033[31m" << "failed";
    exit(1);
  }

  cout << "\033[0m" << "initializing cuda engine ... ";
  engine = builder->buildEngineWithConfig(*network, *config);
  if (!engine) {
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "initializing execution context ... ";
  context = engine->createExecutionContext();
  if (!context) {
    cout << "\033[31m" << "failed" << endl;
    exit(1);
  }
  cout << "\033[32m" << "done" << endl;

  cout << "\033[0m" << "initializing gpu buffer ... ";
  inputDims = engine->getBindingDimensions(0);
  outputDims = engine->getBindingDimensions(1);

  for (size_t i = 0; i < inputDims.nbDims; i++) {
    inputSize *= inputDims.d[i];
  }

  inputSize *= sizeof(float);

  for (size_t i = 0; i < outputDims.nbDims; i++) {
    outputSize *= outputDims.d[i];
  }

  outputSize *= sizeof(float);

  cudaError_t output_malloc = cudaMalloc(&buffers[1], outputSize);
  if (output_malloc != cudaSuccess) {
    cout << "\033[31m" << "failed for output buffer with error " << output_malloc << endl;
    exit(1);
  }

  cudaError_t input_malloc = cudaMalloc(&buffers[0], inputSize);
  if (input_malloc != cudaSuccess) {
    cout << "\033[31m" << "failed for input buffer with error " << input_malloc << endl;
    exit(1);
  }

  cout << "\033[32m" << "done" << endl;
  return inputSize;
}

vector<string> run_inference(int batch_size, void* cpu_buffer) {
  context->enqueue(batch_size, &buffers[0], 0, nullptr);
  cudaDeviceSynchronize();

  vector<float> inference_output(outputSize);
  cudaMemcpy(inference_output.data(), (float *) buffers[1], outputSize, cudaMemcpyDefault);

  for (vector<float>::iterator it = inference_output.begin(); it != inference_output.end(); it++) {
    *it = exp(*it);
  }

  vector< string > fish_classification(batch_size);

  for (int i = 0; i < batch_size; i++) {
    float sum = 0;

    for (int j = 0; j < MODEL_OUTPUT_CLASSES; j++) {
      sum += inference_output[i * MODEL_OUTPUT_CLASSES + j];
    }

    for (int j = 0; j < MODEL_OUTPUT_CLASSES; j++) {
      inference_output[i * MODEL_OUTPUT_CLASSES + j] /= sum;
    }

    int fish_species = 0;
    float max_probability = inference_output[i * MODEL_OUTPUT_CLASSES];

    for (int j = 1; j < MODEL_OUTPUT_CLASSES; j++) {
      if (inference_output[i * MODEL_OUTPUT_CLASSES + j] > max_probability) {
        fish_species = j;
        max_probability = inference_output[i * MODEL_OUTPUT_CLASSES + j];
      }
    }

    if (max_probability >= MINIMUM_INFERENNCE_CONFIDENCE) {
      fish_classification[i] = MODEL_CLASS_LABELS[fish_species];
    } else {
      fish_classification[i] = "none";
    }
    
  }

  return fish_classification;
}
