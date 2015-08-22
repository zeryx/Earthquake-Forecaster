#ifndef DATAARRAYS_H
#define DATAARRAYS_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); cudaDeviceReset(); exit(EXIT_FAILURE);}} while (0)
#endif
template <typename T>
struct kernelArray{
    T* array;
    int size;
};

struct Lock {
    int *mutex;
    Lock( void ) {
        CUDA_SAFE_CALL( cudaMalloc( (void**)&mutex, sizeof(int) ) );
        CUDA_SAFE_CALL( cudaMemset( mutex, 0, sizeof(int) ) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

    __device__ void lock( void ) {
    #if __CUDA_ARCH__ >= 200
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    #endif
    }

    __device__ void unlock( void ) {
    #if __CUDA_ARCH__ >= 200
        atomicExch( mutex, 0 );
    #endif
    }
};



#endif
