#include <kernelDefs.h>

__global__ void normalizeKern(kernelArray<double> vect, kernelArray<int> params, float *avgFitness,  int device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int fitnessval = params.array[19] + idx + device_offset;
    const double fitAvg =(*avgFitness);
    vect.array[fitnessval] = vect.array[fitnessval]/ fitAvg;
    if(vect.array[fitnessval] < 1){
        vect.array[fitnessval] = 0;
    }
}
