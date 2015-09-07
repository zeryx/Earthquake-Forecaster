#include <kernelDefs.h>

__global__ void normalizeKern(kernelArray<float> vect, kernelArray<int> params, float *avgFitness,  size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int fit = params.array[19] + idx + device_offset;
    const int age = params.array[25] + idx + device_offset;
    if(*avgFitness > 0){
        if((vect.array[fit]/(*avgFitness)) < 1){//the value set here dictates "how good" an individual has to be to be eligible to reproduce, 1 better than average.
            vect.array[fit] = 0;
            vect.array[age] = 0; // it is a new born, and has no age.

        }
    }
}
