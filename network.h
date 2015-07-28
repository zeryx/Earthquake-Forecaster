#ifndef NETWORK_H
#define NETWORK_H
#include <thrust/host_vector.h>
#include <map>
#include "individual.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
class  NetworkGenetic{

public:
    NetworkGenetic();
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                   const int &numOutNeurons, std::map<const int, int> &connections);
    bool generatePop(int popsize); // tells the network how many individuals you want to start with
private:
    thrust::host_vector<Individual> _individuals;
    thrust::host_vector<int> _constantNNParams;
    int _neuronsTotalNum;
    std::map<const int, int> _connections;
};


#endif
