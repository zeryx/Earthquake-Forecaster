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
    for(int i=0; i<numWeights; i++){
        seed = idx +i;
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref._array[idx*numWeights+i] = uniDist(randEng);
    }
}

NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                               const int &numOutNeurons, std::map<const int, int> &connections){
    _constantNNParams.push_back(numInNeurons);
    _constantNNParams.push_back(numHiddenNeurons);
    _constantNNParams.push_back(numOutNeurons);
    _neuronsTotalNum = numInNeurons + numHiddenNeurons + numOutNeurons;
    _connections = connections;
}

thrust::device_vector<double> NetworkGenetic::generatePop(int popsize){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocksize; //the blocksize defined by the configurator
    int minGridSize; //the minimum grid size needed to achive max occupancy
    int gridSize; // the actual grid size needed

    thrust::device_vector<double> result(popsize*_neuronsTotalNum);
    long time = std::clock();
    cudaEventRecord(start, 0);
    cudaDeviceSynchronize();
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blocksize, (void*)genWeights<double>, 0, popsize);
    gridSize = (popsize + blocksize -1)/blocksize;
    std::cout<< blocksize << minGridSize << gridSize<<std::endl;
    genWeights<double><<<gridSize, blocksize>>>(convertToKernel<double>(result), time, _neuronsTotalNum);
    cudaDeviceSynchronize();
    float miliseconds = 0;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout<<"total compute time: "<<miliseconds<<" ms"<<std::endl;

    return result;
}
