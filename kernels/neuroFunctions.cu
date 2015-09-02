#include <neuroFunc.h>
#include <utilFunc.h>

__host__ __device__ void neuroSum(double &store, double &input){
    store += input;
}

__host__ __device__ void neuroZero(double &store){
    store =0;
}

__host__ __device__ void neuroSquash(double &store){
    store = ActFunc(store);
}

__host__ __device__ void neuroMemGate(double &memIn, double &input, double &output, float min){
    if(memIn > min)
        neuroSum(output, input);
}
__host__ __device__ void neuroMemForget(double &memForget, double &mem, float min){
    if(memForget > min)
        neuroZero(mem);

}
