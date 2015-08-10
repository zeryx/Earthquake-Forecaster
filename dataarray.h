#ifndef dataArray_H
#define dataArray_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
template <typename T>
struct dataArray{
    thrust::device_ptr<T> array;
    int size;
};
template <typename T>
struct unifiedArray{
    T* array;
    int size;
};

template <typename T>
dataArray<T> convertToKernel(thrust::device_vector<T> &dVect){
    dataArray<T> kArray;
    kArray.array = thrust::device_pointer_cast(dVect.data());
    kArray.size  = (int) dVect.size();
    return kArray;
}

template <typename T>
dataArray<T> convertToKernel(thrust::device_vector<T> *dVect){
    dataArray<T> kArray;
    kArray.array = thrust::device_pointer_cast(dVect->data());
    kArray.size = (int) dVect->size();
    return kArray;
}

#endif
