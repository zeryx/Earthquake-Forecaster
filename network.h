#ifndef NETWORK_H
#define NETWORK_H
#include "dataarray.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                   const int &numOutNeurons, const std::map<const int, int> &connections);
    void errorFunc();
    void initializeWeights(); //initializes _data array and fills with random numbers
    void loadFromFile(std::string file);
    void allocateHostAndGPUObjects(float hostMemory, float deviceMemory,
                                   std::map<const std::string, float> pHostRam,  std::map<const std::string, float> pDeviceRam);
private:
    thrust::device_vector<int> _NNParams;//1.

    thrust::device_vector<double> _DGeneticsData; //device loaded memory object containing the training weights & fitness data.
    thrust::device_vector<int> _DInputData; //device loaded memory object containing the site input data.
    thrust::device_vector<double> _DTrainingData; //device loaded memory object containing the teacher data.
    thrust::device_vector<double> _DPMAIndex; //device loaded memory object containing the planetary magnetic activity index. (all of it should fit)
    thrust::host_vector<double> _HGeneticsData; //overflow of Genetics Data
    thrust::host_vector<int> _HInputData; //overflow of input Data that cannot fit in GPU memory due to resource limitations.
    thrust::host_vector<double> _HTrainingData; //overflow of training data
    int _neuronsTotalNum, _initialPopsize;
    std::map<const int, int> _connections;
};


#endif
