#ifndef DATAARRAY_H
#define DATAARRAY_H
#include <thrust/device_vector.h>
template <typename T>
struct DataArray{
    T *_array;
    int _size;
};

template <typename T>
DataArray<T> convertToKernel(thrust::device_vector<T>& dVect){
    DataArray<T> kArray;
    kArray._array = thrust::raw_pointer_cast(dVect.data());
    kArray._size  = (int) dVect.size();
    return kArray;
}

#endif
