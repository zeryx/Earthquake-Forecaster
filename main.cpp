#include "getsys.h"
#include "network.h"
#include <iostream>
#include <map>


int main(void){
    int inputs = 2; // half of the gpu contains training data, the other half contains
    int hidden = 4;
    int outputs = 1;
    int mem = GetRamInKB();
    std::cout<<"total ram in KB: "<<mem<<std::endl;
    int popcount = 1e6; // max size is = 50% of ram, remainder is training data & input data
    std::map<const int, int> connections;
    NetworkGenetic ConstructedNetwork(inputs, hidden, outputs, connections, popcount);
    ConstructedNetwork.initializeWeights();
    return 0;
}
