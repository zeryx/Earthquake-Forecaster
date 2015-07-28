
#include "network.h"
#include <iostream>
#include <map>

int main(void){
    int inputs = 2;
    int hidden = 4;
    int outputs = 1;
    std::map<int, int> connections;
    connections[0] = 3;
    connections[3] = 4;
    NetworkGenetic ConstructedNetwork(inputs, hidden, outputs, connections);
    ConstructedNetwork.generatePop(50000);

    return 0;
}
