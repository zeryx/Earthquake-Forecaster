#include "network.h"
#include <iostream>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/swap.h>
#include <vector>
#include <thrust/system_error.h>
struct genRand{
    int numofNeurons;

    __host__ __device__
    genRand(int _numofNeurons) : numofNeurons(_numofNeurons){};

    __host__ __device__
    Individual operator() (int idx) {
        Individual thisPerson;
        for(int i=0; i<numofNeurons; i++){
            thrust::default_random_engine randEng;
            thrust::uniform_real_distribution<float> uniDist;
            thisPerson._weights.push_back(uniDist(randEng));
            randEng.discard(idx);
        }
        return thisPerson;
    }
};

NetworkGenetic::NetworkGenetic(){}

NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                               const int &numOutNeurons, std::map<int, int> &connections){
    _constantNNParams.push_back(numInNeurons);
    _constantNNParams.push_back(numHiddenNeurons);
    _constantNNParams.push_back(numOutNeurons);
    _neuronsTotalNum = numInNeurons + numHiddenNeurons + numOutNeurons;
    _connections = connections;
}

bool NetworkGenetic::generatePop(int popsize){
    _individuals.resize(popsize);
    try{
        thrust::transform( thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(popsize),
                           _individuals.begin(), genRand(_neuronsTotalNum));
    }
    catch(thrust::system_error &err){
        std::cerr<<"error transforming: "<<err.what()<<std::endl;
        return false;
    }
    for(int i=0; i<_individuals.size(); i++){
        std::cout<<"i: "<< i << std::endl;
        for(int k=0; k<_individuals[i]._weights.size(); k++){
            std::cout<<_individuals[i]._weights[k]<<std::endl;
        }
    }
    return true;
}
