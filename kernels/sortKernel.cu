#include <kernelDefs.h>

__global__ void sortKern(kernelArray<std::pair<int, double> > vec, int j, int k, int ind_offset){
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        if ((i&k)==0) {
            /* Sort ascending */
            if (vec.array[i+ind_offset].second>vec.array[ixj+ind_offset].second) {
                /* exchange(i,ixj); */
                std::pair<int, double> temp = vec.array[i+ind_offset];
                vec.array[i+ind_offset].first = vec.array[ixj+ind_offset].first;
                vec.array[i+ind_offset].second = vec.array[ixj+ind_offset].second;
                vec.array[ixj+ind_offset].first = temp.first;
                vec.array[ixj+ind_offset].second = temp.second;
            }
        }
        if ((i&k)!=0) {
            /* Sort descending */
            if (vec.array[i+ind_offset].second<vec.array[ixj+ind_offset].second) {
                /* exchange(i,ixj); */
                std::pair<int, double> temp = vec.array[i+ind_offset];
                vec.array[i+ind_offset].first = vec.array[ixj+ind_offset].first;
                vec.array[i+ind_offset].second = vec.array[ixj+ind_offset].second;
                vec.array[ixj+ind_offset].first = temp.first;
                vec.array[ixj+ind_offset].second = temp.second;
            }
        }
    }

}
