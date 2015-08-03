#ifndef dataArray_H
#define dataArray_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
template <typename T>
struct dataArray{
    thrust::device_ptr<T> _array;
    int _size;
};

template <typename T>
class hVector{
public:
    void setMax(long long maxLen){
        _hVect.resize(maxLen);
        _itr = 0;
        _maxLen = maxLen;
    }

public:
    thrust::host_vector<T> _hVect;
    long long _itr;
    long long _maxLen;
};

template <typename T>
dataArray<T> convertToKernel(thrust::device_vector<T> &dVect){
    dataArray<T> kArray;
    kArray._array = thrust::device_pointer_cast(dVect.data());
    kArray._size  = (int) dVect.size();
    return kArray;
}

#endif
