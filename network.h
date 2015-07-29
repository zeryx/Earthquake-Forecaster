#ifndef NETWORK_H
#define NETWORK_H
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <map>

struct Individual{//stores the weights and its fitness values
public:
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
    thrust::device_vector<Individual> _individuals;
    thrust::host_vector<int> _constantNNParams;
    int _neuronsTotalNum;
    std::map<const int, int> _connections;
};


#endif
