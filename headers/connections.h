#ifndef CONNECTIONS_H
#define CONNECTIONS_H
#include <string>
enum neuroType{
    typeNULL =0,
    typeInput =1,
    typeHidden =2,
    typeMemory = 3,
    typeMemGateIn = 4,
    typeMemGateOut = 5,
    typeMemGateForget = 6,
    typeOutput = 7,
    typeSquash =8,
    typeZero =9,
    typeBias =10
};

template <typename T, typename H>
struct descriptor{
    T def;
    H id;
};

struct Order{
    descriptor<neuroType, int> first;
    descriptor<neuroType, int> second;
    descriptor<neuroType, int> third;
};




#endif
