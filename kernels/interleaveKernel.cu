#include <kernelDefs.h>
__constant__ int trainingsize;

__global__ void interKern(kernelArray<int> in, kernelArray<int> out, int* site_offset, int* channel_offset, int sampleRate, int numOfSites){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;// each thread manages a single second for the 3d flattened input array.
    for(int k=0; k<numOfSites; k++){
        int meanch1 = 0, meanch2 =0, meanch3 = 0;
        for(int i=0; i<sampleRate*3600/trainingsize; i++){
            meanch1 += in.array[k*sampleRate*3600*3 + 0*sampleRate*3600 + idx+i];
            meanch2 += in.array[k*sampleRate*3600*3 + 1*sampleRate*3600 + idx+i];
            meanch3 += in.array[k*sampleRate*3600*3 + 2*sampleRate*3600 + idx+i];
        }
        meanch1 = meanch1/sampleRate;
        meanch2 = meanch2/sampleRate;
        meanch3 = meanch3/sampleRate;

        out.array[site_offset[k]+channel_offset[0]+idx] = meanch1;
        out.array[site_offset[k]+channel_offset[1]+idx] = meanch2;
        out.array[site_offset[k]+channel_offset[2]+idx] = meanch3;
    }
}
