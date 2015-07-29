
#include "network.h"
#include <iostream>
#include <map>

int main(void){
    int inputs = 2;
    int hidden = 4;
    int outputs = 1;
    std::map<const int, int> connections;
    NetworkGenetic ConstructedNetwork(inputs, hidden, outputs, connections);
    ConstructedNetwork.generatePop(15000);

    return 0;
}
