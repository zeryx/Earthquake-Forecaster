#include <kernelDefs.h>

//using
extern __constant__ int params[];
//endofusing

__global__ void bitonicSortKern(kernelArray<double> Vec, int j, int k, size_t device_offset){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int first=idx;
    const int second = first^j;
    const int fitnessOffset = params[19] + device_offset;
    const int wtOffset = params[11] + device_offset;
    const int ind = params[10];
    /* The threads with the lowest ids sort the array. */
    if ((second)>first){
        if ((first&k)!=0) {
            /* Sort Decending */
            if (Vec.array[first+fitnessOffset]>Vec.array[second+fitnessOffset]) {
                /* exchange(first,second); */
                double temp = Vec.array[first+fitnessOffset];
                Vec.array[first+fitnessOffset] = Vec.array[second+fitnessOffset];
                Vec.array[second+fitnessOffset] = temp;
                for(int n=0; n<params[1]; n++){
                    double temp_wt = Vec.array[wtOffset + first + n*ind];
                    Vec.array[wtOffset+first+n*ind] = Vec.array[wtOffset+second+n*ind];
                    Vec.array[wtOffset+second+n*ind] = temp_wt;
                }
            }
        }
        if ((first&k)==0) {
            /* Sort Ascending */
            if (Vec.array[first+fitnessOffset]<Vec.array[second+fitnessOffset]) {
                /* exchange(first,second); */
                double temp = Vec.array[first+fitnessOffset];
                Vec.array[first+fitnessOffset] = Vec.array[second+fitnessOffset];
                Vec.array[second+fitnessOffset] = temp;
                for(int n=0; n<params[1]; n++){
                    double temp_wt = Vec.array[wtOffset + first + n*ind];
                    Vec.array[wtOffset+first+n*ind] = Vec.array[wtOffset+second+n*ind];
                    Vec.array[wtOffset+second+n*ind] = temp_wt;
                }
            }
        }
    }
}
