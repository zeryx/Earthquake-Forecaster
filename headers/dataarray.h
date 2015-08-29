#ifndef DATAARRAYS_H
#define DATAARRAYS_H
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); cudaDeviceReset(); exit(EXIT_FAILURE);}} while (0)
#endif
template <typename T>
struct kernelArray{
    T* array;
    int size;
};



#endif
