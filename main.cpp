#include "network.h"
#include <iostream>
#include <map>


int main(int argc, char** arg){
    int inputs = 8;
    int hidden = 4;
    int memory = 2; //LSTM neurons
    int hidden_layers = 2;
    int outputs = 3;
    thrust::pair<int, int> connections; // I'll actually populate this at some point
    std::map<const std::string, float> hostRamPercentageMap, deviceRamPercentageMap;
    hostRamPercentageMap["genetics"] = 0.40;
    hostRamPercentageMap["input & training"] = 0.60; // the bulk of the host memory should contain input & training data.
    deviceRamPercentageMap["genetics"] = 0.80; //the bulk of the GPU should contain the genetics data
    deviceRamPercentageMap["input & training"] = 0.20;
    NetworkGenetic ConstructedNetwork(inputs, hidden, memory, outputs, hidden_layers,  connections);
    ConstructedNetwork.allocateHostAndGPUObjects(hostRamPercentageMap, deviceRamPercentageMap,0.25, 0.85);
    ConstructedNetwork.getTestInfo("../mount/data");
    ConstructedNetwork.initializeWeights();
    std::cin.get();
    return 0;
}
