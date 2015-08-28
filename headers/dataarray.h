#ifndef DATAARRAYS_H
#define DATAARRAYS_H
//simple struct that contains both a pointer of type T and a size parameter, used within kernels.
template <typename T>
struct kernelArray{
    T* array;
    int size;
};



#endif
