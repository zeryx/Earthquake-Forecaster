
#include "network.h"
#include <iostream>
#include <map>
#include <cuda.h>
int main(void){
    int inputs = 2;
    int hidden = 4;
    int outputs = 1;
    int popcount = 150000;
    std::map<const int, int> connections;
    NetworkGenetic ConstructedNetwork(inputs, hidden, outputs, connections);
    thrust::device_vector<double> trainingWeights = ConstructedNetwork.generatePop(popcount);
    return 0;
}
