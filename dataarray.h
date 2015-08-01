#ifndef dataArray_H
#define dataArray_H
#include <thrust/device_vector.h>
template <typename T>
struct dataArray{
    thrust::device_ptr<T> _array;
    int _size;
};

template <typename T>
class hVector{
public:
    void setMax(unsigned int maxLen){
        _dVect.resize(maxLen);
        _itr = 0;
        _maxLen = maxLen;
    }
public:
    thrust::device_vector<T> _dVect;
    int _itr;
    int _maxLen;
};

template <typename T>
dataArray<T> convertToKernel(thrust::device_vector<T>& dVect){
    dataArray<T> kArray;
    kArray._array = thrust::device_pointer_cast(dVect.data());
    kArray._size  = (int) dVect.size();
    return kArray;
}

#endif
