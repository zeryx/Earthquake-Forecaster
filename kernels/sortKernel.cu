#include <kernelDefs.h>

__global__ void bitonicSortKern(kernelArray<float> vec, kernelArray<int> params, int j, int k, size_t device_offset){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int first=idx;
    const int second = first^j;
    /* it's important to pass identifying and important data between the individuals, so nothing gets lost during soring.
     * the important attached data is the weigts, memory, fitness, community magnitude, and age */
    const int wtOffset = params.array[11] + device_offset;
    const int mem_offset = params.array[14] + device_offset;
    const int fitOffset= params.array[19] + device_offset;
    const int magOffset = params.array[20] + device_offset;
    const int ageOffset = params.array[25] + device_offset;
    const int ind = params.array[10];
    /* The threads with the lowest ids sort the array. */
    if ((second)>first){
        if ((first&k)!=0) {
            /* Sort Decending */
            if (vec.array[first+fitOffset]>vec.array[second+fitOffset]) {
                /* exchange(first,second); */
                //pass fitness
                float tmp_fit = vec.array[first+fitOffset];
                vec.array[first+fitOffset] = vec.array[second+fitOffset];
                vec.array[second+fitOffset] = tmp_fit;
                //pass age
                float tmp_age = vec.array[first +ageOffset];
                vec.array[first+ageOffset] = vec.array[second+ageOffset];
                vec.array[second+ageOffset] = tmp_age;
                //pass communityMag
                for(int n=0; n<params.array[23]; n++){
                    float tmp_mag = vec.array[magOffset + first + n*ind];
                    vec.array[magOffset+first+n*ind] = vec.array[magOffset+second+n*ind];
                    vec.array[magOffset+second+n*ind] = tmp_mag;
                }
                //pass memory
                for(int n=0; n<params.array[5]; n++){
                    float tmp_mem = vec.array[mem_offset + first + n*ind];
                    vec.array[mem_offset+first+n*ind] = vec.array[mem_offset+second+n*ind];
                    vec.array[mem_offset+second+n*ind] = tmp_mem;
                }
                //pass weights
                for(int n=0; n<params.array[1]; n++){
                    float tmp_wt = vec.array[wtOffset + first + n*ind];
                    vec.array[wtOffset+first+n*ind] = vec.array[wtOffset+second+n*ind];
                    vec.array[wtOffset+second+n*ind] = tmp_wt;
                }
            }
        }
        if ((first&k)==0) {
            /* Sort Ascending */
            if (vec.array[first+fitOffset]<vec.array[second+fitOffset]) {
                /* exchange(first,second); */
                //pass fitness
                float tmp_fit = vec.array[first+fitOffset];
                vec.array[first+fitOffset] = vec.array[second+fitOffset];
                vec.array[second+fitOffset] = tmp_fit;
                //pass age
                float tmp_age = vec.array[first +ageOffset];
                vec.array[first+ageOffset] = vec.array[second+ageOffset];
                vec.array[second+ageOffset] = tmp_age;
                //pass communityMag
                for(int n=0; n<params.array[23]; n++){
                    float tmp_mag = vec.array[magOffset + first + n*ind];
                    vec.array[magOffset+first+n*ind] = vec.array[magOffset+second+n*ind];
                    vec.array[magOffset+second+n*ind] = tmp_mag;
                }
                //pass memory
                for(int n=0; n<params.array[5]; n++){
                    float tmp_mem = vec.array[mem_offset + first + n*ind];
                    vec.array[mem_offset+first+n*ind] = vec.array[mem_offset+second+n*ind];
                    vec.array[mem_offset+second+n*ind] = tmp_mem;
                }
                //pass weights
                for(int n=0; n<params.array[1]; n++){
                    float tmp_wt = vec.array[wtOffset + first + n*ind];
                    vec.array[wtOffset+first+n*ind] = vec.array[wtOffset+second+n*ind];
                    vec.array[wtOffset+second+n*ind] = tmp_wt;
                }
            }
        }
    }
}

