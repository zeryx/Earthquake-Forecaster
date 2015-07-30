#include "getsys.h"
#include "network.h"
#include <iostream>
#include <map>


int main(void){
    int inputs = 2;
    int hidden = 4;
    int memory = 1; //SFTM neurons
    int outputs = 1;
    std::map<const int, int> connections; // I'll actually populate this at some point
    unsigned int hostMem = GetHostRamInBytes()*0.75; //make a host memory container, this is the max
    unsigned int deviceMem = GetDeviceRamInBytes()*0.85; //dito for gpu
    std::map<const std::string, float> hostRamPercentageMap, deviceRamPercentageMap;
    hostRamPercentageMap["genetics"] = 0.25;
    hostRamPercentageMap["input & training"] = 0.75; // the bulk of the host memory should contain input & training data.
    deviceRamPercentageMap["genetics"] = 0.75; //the bulk of the GPU should contain the genetics data
    deviceRamPercentageMap["input & training"] = 0.25;
    NetworkGenetic ConstructedNetwork(inputs, hidden, memory, outputs, connections);
    ConstructedNetwork.allocateHostAndGPUObjects(hostMem, deviceMem, hostRamPercentageMap, deviceRamPercentageMap);
    ConstructedNetwork.initializeWeights();
    return 0;
}
