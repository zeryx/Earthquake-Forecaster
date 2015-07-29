#ifndef NETWORK_H
#define NETWORK_H
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>
#include <cuda.h>
#include <cuda_runtime_api.h>

class Individual{//stores the weights and its fitness values
public:
    Individual(int numWeights){cudaMalloc((void**) &_weights, numWeights*sizeof(float));}
    float* _weights;
    float _absoluteFitness, _relativeFitness;
};

class  NetworkGenetic{

public:
    NetworkGenetic();
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                   const int &numOutNeurons, std::map<const int, int> &connections);
    bool generatePop(int popsize); // tells the network how many individuals you want to start with
private:
    thrust::host_vector<int> _constantNNParams;
    int _neuronsTotalNum;
    std::map<const int, int> _connections;
};


#endif
