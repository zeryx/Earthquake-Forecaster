#include "getsys.h"
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
    unsigned int hostMem = GetHostRamInBytes()*0.75; //make a host memory container, this is the max
    unsigned int deviceMem = GetDeviceRamInBytes()*0.90; //dito for gpu
    std::map<const std::string, float> hostRamPercentageMap, deviceRamPercentageMap;
    hostRamPercentageMap["genetics"] = 0.25;
    hostRamPercentageMap["input & training"] = 0.75; // the bulk of the host memory should contain input & training data.
    deviceRamPercentageMap["genetics"] = 0.80; //the bulk of the GPU should contain the genetics data
    deviceRamPercentageMap["input & training"] = 0.20;
    NetworkGenetic ConstructedNetwork(inputs, hidden, memory, outputs, hidden_layers,  connections);
    ConstructedNetwork.importSitesData("../mount/data/105/SiteInfo.xml");
    ConstructedNetwork.importKpData("../mount/data/105/Kp.xml");
    ConstructedNetwork.allocateHostAndGPUObjects(hostMem, deviceMem, hostRamPercentageMap, deviceRamPercentageMap);
    ConstructedNetwork.initializeWeights();
    return 0;
}
