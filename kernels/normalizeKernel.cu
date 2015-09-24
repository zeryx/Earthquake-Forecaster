#include <kernelDefs.h>

__global__ void normalizeKern(kernelArray<double> Vec, kernelArray<int> params, double *avgFitness,  size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int ind = params.array[10];
    const int fit = params.array[19] + idx + device_offset;
    const int mem = params.array[14] + idx + device_offset;
    const int communityMag = params.array[20] + idx + device_offset;
    if(*avgFitness > 0){
        if((Vec.array[fit]/(*avgFitness)) < 1){//the value set here dictates "how good" an individual has to be to be eligible to reproduce, 1 better than average.
            Vec.array[fit] = 0;

        }
    }

    //reset community magnitude & memory values for every individual, since the next run is a brand new trial.
    for(int i=0; i<params.array[23]; i++){
         Vec.array[communityMag + i*ind] = 1;
    }

    for(int i=0; i<params.array[5]; i++)
        Vec.array[mem + i*ind] = 0;
}
