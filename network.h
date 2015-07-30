#ifndef NETWORK_H
#define NETWORK_H
#include "dataarray.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>
#include <cuda.h>
#include <cuda_runtime_api.h>

class  NetworkGenetic{
public:
    NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons,
                   const int &numOutNeurons, std::map<const int, int> &connections);
    thrust::device_vector<double> generatePop(int popsize); // tells the network how many individuals you want to start with
private:
    thrust::device_vector<int> _constantNNParams;
    int _neuronsTotalNum;
    std::map<const int, int> _connections;
};


#endif
