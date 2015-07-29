#include "network.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system_error.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

struct genRand{
 __host__ __device__
 float operator()(const unsigned int n) const{
    double result;
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<float> uniDist(0,1);
    randEng.discard(n);
    result = uniDist(randEng);
    return result;
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

    thrust::device_vector<float> testing(popsize);
    try{
        cudaEventRecord(start);
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(popsize), testing.begin(), genRand());
    }
    catch(thrust::system_error &err){
        std::cerr<<"error transforming: "<<err.what()<<std::endl;
        return false;
    }
cudaEventRecord(stop);
float miliseconds = 0;
    for(int i=0; i<popsize; i++){
        std::cout<<testing[i]<<std::endl;
    }
    cudaEventElapsedTime(&miliseconds, start, stop);

    return true;
}
