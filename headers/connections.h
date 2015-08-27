#ifndef CONNECTIONS_H
#define CONNECTIONS_H
#include <string>
enum neuronType{
    typeInput =1,
    typeHidden =2,
    typeMemory = 3,
    typeMemGateIn = 4,
    typeMemGateOut = 5,
    typeMemGateForget = 6,
    typeOutput = 7
};

template <typename T, typename H>
struct devicePair{
    T first;
    H second;
};
typedef std::pair<neuronType, int> hcon;
typedef devicePair<neuronType, int>dcon;



#endif
