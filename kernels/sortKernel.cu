#include <kernelDefs.h>

__global__ void sortKern(kernelArray<double> vec, kernelArray<int> params, int j, int k, size_t device_offset){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int first=idx;
    const int second = first^j;
    const int fitnessOffset = params.array[19] + device_offset;
    const int wtOffset = params.array[11] + device_offset;
    const int ind = params.array[10];
    /* The threads with the lowest ids sort the array. */
    if ((second)>first){
        if ((first&k)==0) {
            /* Sort Decending */
            if (vec.array[first+fitnessOffset]<vec.array[second+fitnessOffset]) {
                /* exchange(first,second); */
                double temp = vec.array[first+fitnessOffset];
                vec.array[first+fitnessOffset] = vec.array[second+fitnessOffset];
                vec.array[second+fitnessOffset] = temp;
                for(int n=0; n<params.array[1]; n++){
                    double temp_wt = vec.array[wtOffset + first + n*ind];
                    vec.array[wtOffset+first+n*ind] = vec.array[wtOffset+second+n*ind];
                    vec.array[wtOffset+second+n*ind] = temp_wt;
                }
            }
        }
        else if ((first&k)!=0) {
            /* Sort Ascending */
            if (vec.array[first+fitnessOffset]>vec.array[second+fitnessOffset]) {
                /* exchange(first,second); */
                double temp = vec.array[first+fitnessOffset];
                vec.array[first+fitnessOffset] = vec.array[second+fitnessOffset];
                vec.array[second+fitnessOffset] = temp;
                for(int n=0; n<params.array[1]; n++){
                    double temp_wt = vec.array[wtOffset + first + n*ind];
                    vec.array[wtOffset+first+n*ind] = vec.array[wtOffset+second+n*ind];
                    vec.array[wtOffset+second+n*ind] = temp_wt;
                }
            }
        }
    }
}
