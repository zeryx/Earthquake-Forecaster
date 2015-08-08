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
class hVector{
public:
    void setMax(long long maxLen){
        delete _hVect;
        _hVect = new thrust::host_vector<double>(maxLen);
        _itr = 0;
        _maxLen = maxLen;
    }
    ~hVector(){
        delete _hVect;
    }

public:
    thrust::host_vector<T>* _hVect;
    long long _itr;
    long long _maxLen;
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
