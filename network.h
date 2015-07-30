#ifndef NETWORK_H
#define NETWORK_H
#include "dataarray.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>
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
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                   const int &numOutNeurons, const std::map<const int, int> &connections, const int popsize);
    void errorFunc();
    void initializeWeights(); //initializes _data array and fills with random numbers
private:
    thrust::device_vector<int> _constantNNParams;//constant parameters for error function connections & other utility paramters.

    thrust::device_vector<double> _DGeneticsData; //device loaded memory object containing the training weights & fitness data.
    thrust::device_vector<int> _DInputData; //device loaded memory object containing the site input data.
    thrust::device_vector<double> _DTrainingData; //device loaded memory object containing the teacher data.
    thrust::device_vector<double> _DPMAIndex; //device loaded memory object containing the planetary magnetic activity index.
    thrust::device_vector<double> STLM_MEMORY;
    thrust::host_vector<double> _HGeneticsData; //overflow of Genetics Data
    thrust::host_vector<int> _HInputData; //overflow of input Data that cannot fit in GPU memory due to resource limitations.
    thrust::host_vector<double> _HTrainingData; //overflow of training data
    thrust::host_vector<double> _HPMAIndex; //overflow of PMA index
    int _neuronsTotalNum, _popsize;
    std::map<const int, int> _connections;
};


#endif
