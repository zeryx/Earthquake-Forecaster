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
typedef std::pair<neuronType, int> con;

//template <typename T, typename H>
//struct mypair{
//    T first;
//    H second;
//};

//struct con{
//    neuronType first;
//    int second;
//};

#endif
