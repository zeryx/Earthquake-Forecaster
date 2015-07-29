#include "network.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system_error.h>
#include <cuda.h>
#include <vector>
#include <cuda_runtime_api.h>

struct genRand: thrust::unary_function<Individual, int>{

    int numWeights;

    genRand(int _numWeights) : numWeights(_numWeights){}

    __host__ __device__
    Individual operator()(Individual n) const{
        unsigned int idx= threadIdx.x*blockDim.x;

        n._weights = new float[numWeights];

        for(int i=0; i<numWeights; i++){
            idx = idx +i;
            thrust::default_random_engine randEng;
            thrust::uniform_real_distribution<float> uniDist(0,1);
            randEng.discard(idx);
            n._weights[i] =  uniDist(randEng);
        }
        return n;
    }};


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
    thrust::device_vector<Individual> testing;
    testing.resize(popsize);

    try{
        cudaEventRecord(start);
        thrust::transform(testing.begin(),
                          testing.end(), testing.begin(), genRand(_neuronsTotalNum));
    }
    catch(thrust::system_error &err){
        std::cerr<<"error transforming: "<<err.what()<<std::endl;
        return false;
    }
    cudaEventRecord(stop);
    float miliseconds = 0;
    std::vector<Individual> printer(testing.size());
    for(int i=0; i<popsize; i++){
        thrust::copy(testing.begin(), testing.end(),printer.begin());
        }
    for(int i=0; i<popsize; i++){
        for(int k=0; k<_neuronsTotalNum; k++){
            std::cout<< printer[k]._weights[k]<<std::endl;
        }
    }
    cudaEventElapsedTime(&miliseconds, start, stop);

    return true;
}
