#include "network.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system_error.h>
#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>

struct genWeights: thrust::unary_function<Individual, int>{

    int numWeights;

    genWeights(int _numWeights) : numWeights(_numWeights){
    }

    __device__
    Individual operator()(Individual n) const{
        unsigned int idx= threadIdx.x*blockDim.x;
        float *weights = new float[numWeights];
        n._weights = weights;
        for(int i=0; i<numWeights; i++){
            idx = idx +i;
            thrust::default_random_engine randEng;
            thrust::uniform_real_distribution<float> uniDist(0,1);
            randEng.discard(idx);
            n._weights[i] =  uniDist(randEng);
        }
        return n;
    }};

struct delWeights{
    __device__ __host__
    Individual operator()(Individual n) const{
        delete[] n._weights;
        return n;
    }
};

NetworkGenetic::NetworkGenetic(){}

NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                               const int &numOutNeurons, std::map<const int, int> &connections){
    _constantNNParams.push_back(numInNeurons);
    _constantNNParams.push_back(numHiddenNeurons);
    _constantNNParams.push_back(numOutNeurons);
    _neuronsTotalNum = numInNeurons + numHiddenNeurons + numOutNeurons;
    _connections = connections;
}

bool NetworkGenetic::generatePop(int popsize){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    thrust::device_vector<Individual> testing(popsize);
    try{
        cudaEventRecord(start, 0);
        thrust::transform(testing.begin(),
                          testing.end(), testing.begin(), genWeights(_neuronsTotalNum));
    }
    catch(thrust::system_error &err){
        std::cerr<<"error transforming: "<<err.what()<<std::endl;
        return false;
    }
    cudaDeviceSynchronize();
    thrust::transform(testing.begin(), testing.end(), testing.begin(), delWeights());
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    float miliseconds = 0;
    cudaError_t err = cudaEventElapsedTime(&miliseconds, start, stop);
    if(err != cudaSuccess){
        std::cout<<"\n\n 1. Error: "<<cudaGetErrorString(err)<<std::endl<<std::endl;
    }
    std::cout<<miliseconds<<" ms"<<std::endl;

    return true;
}
