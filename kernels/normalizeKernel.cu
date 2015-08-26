#include <kernelDefs.h>

//using
extern __constant__ int params[];
//endofusing

__global__ void normalizeKern(kernelArray<double> Vec, double *avgFitness,  size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int fitnessval = params[19] + idx + device_offset;
    Vec.array[fitnessval] = Vec.array[fitnessval]/(*avgFitness);
    if(Vec.array[fitnessval] < 1.07){//the value set here dictates "how good" an individual has to be to be eligible to reproduce, 1 better than average.
        Vec.array[fitnessval] = 0;
    }
}
