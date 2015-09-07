#include <neuroFunc.h>
#include <utilFunc.h>

__host__ __device__ void neuroSum(float &store, float &input){
    store += input;
}

__host__ __device__ void neuroZero(float &store){
    store =0;
}

__host__ __device__ void neuroSquash(float &store){
    store = ActFunc(store);
}

__host__ __device__ void neuroMemGate(float &memIn, float &input, float &output, float min){
    if(memIn > min || memIn < -min)
        neuroSum(output, input);
}
__host__ __device__ void neuroMemForget(float &memForget, float &mem, float min){
    if(memForget > min || memForget < -min)
        neuroZero(mem);

}
