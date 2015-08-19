#ifndef DATAARRAYS_H
#define DATAARRAYS_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
template <typename T>
struct kernelArray{
    T* array;
    int size;
};

#endif
