#include <neuroFunc.h>
#include <utilFunc.h>

__host__ __device__ void neuroSum(double &store, double &input){
    store += input;
}

__host__ __device__ void neuroMulti(double &store, double &input){
    store *= input;
}

__host__ __device__ void neuroZero(double &store){
    store =0;
}

__host__ __device__ void neuroSquash(double &store){
    store = ActFunc(store);
}

__host__ __device__ void neuroMemGateOut(double &memGate, double &input, double &output){

    double tmp = shift(memGate, 1, -1, 1, 0);
    double in = ActFunc(input)*tmp;
    neuroSum(output, in);
}

__host__ __device__ void neuroMemGateIn(double &memGate, double &input, double &output){

    double tmp = shift(memGate, 1, -1, 1, 0);
    double in= input*tmp;
    neuroSum(output, in);
}

__host__ __device__ void neuroMemGateForget(double &memForget, double &mem){
    double tmp = shift(memForget, 1, -1, 1, 0);
    neuroMulti(mem, tmp);
}
