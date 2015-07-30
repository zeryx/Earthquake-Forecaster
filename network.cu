#include "network.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <ctime>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>

template <typename T>
__global__ void genWeights( DataArray<T> ref, long in, int numWeights){
    long idx = blockDim.x*blockIdx.x + threadIdx.x;
    long seed= idx+in;
    thrust::default_random_engine randEng;
    for(int i=0; i<numWeights; i++){
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref._array[idx*numWeights+i] = uniDist(randEng);
    }
}

NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                               const int &numOutNeurons,  const std::map<const int, int> &connections, const int popsize){
    _constantNNParams.push_back(numInNeurons);
    _constantNNParams.push_back(numHiddenNeurons);
    _constantNNParams.push_back(numOutNeurons);
    _neuronsTotalNum = numInNeurons + numHiddenNeurons + numOutNeurons;
    _connections = connections;
    _popsize = popsize;
}

void NetworkGenetic::initializeWeights(){
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL (cudaEventCreate(&start));
    CUDA_SAFE_CALL (cudaEventCreate(&stop));
    int blocksize; //the blocksize defined by the configurator
    int minGridSize; //the minimum grid size needed to achive max occupancy
    int gridSize; // the actual grid size needed
    this->_DGeneticsData.resize(_popsize*(_neuronsTotalNum+1)); // stores the # of individuals to start, which contain their total neuron count + 1 performance double.
    long time = std::clock();
    cudaEventRecord(start, 0);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL (cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blocksize, (void*)genWeights<double>, 0, _popsize));
    gridSize = (_popsize + blocksize -1)/blocksize;
    genWeights<double><<<gridSize, blocksize>>>(convertToKernel<double>(_DGeneticsData), time, _neuronsTotalNum);
    cudaDeviceSynchronize();
    float miliseconds = 0;
    CUDA_SAFE_CALL (cudaEventRecord(stop, 0));
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL (cudaEventElapsedTime(&miliseconds, start, stop));
    std::cout<<"weight generation: total compute time: "<<miliseconds<<" ms"<<std::endl;
    std::cout<<"effective bandwidth (GB/s) : "<<_popsize*_neuronsTotalNum*8/miliseconds/1e9<<std::endl;
}

void NetworkGenetic::errorFunc(){
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    int blocksize; //the blocksize defined by the configurator
//    int minGridSize; //the minimum grid size needed to achive max occupancy
//    int gridSize; // the actual grid size needed
}
