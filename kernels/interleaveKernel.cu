#include <kernelDefs.h>

__global__ void interKern(kernelArray<int> in, kernelArray<int> out,int sampleRate, int numOfSites){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;// each thread manages a single time step for the 3d flattened input array.
    int channel_offset[3]; // 3 incomming data channels
    int sizeofSite = 3600*sampleRate*3;
    int site_offset[20];
    channel_offset[0] = 0;
    channel_offset[1] = 3600*sampleRate; //size of each channel in a site
    channel_offset[2] = 3600*sampleRate + channel_offset[1];
    site_offset[0]=0;
    for(int i=1; i<numOfSites; i++){
        site_offset[i] = sizeofSite + site_offset[i-1];
    }
    for(int k=0; k<numOfSites; k++){
        out.array[site_offset[k]+channel_offset[0]+idx] = in.array[k*sampleRate*3600*3 + 0*sampleRate*3600 + idx];
        out.array[site_offset[k]+channel_offset[1]+idx] = in.array[k*sampleRate*3600*3 + 1*sampleRate*3600 + idx];
        out.array[site_offset[k]+channel_offset[2]+idx] = in.array[k*sampleRate*3600*3 + 2*sampleRate*3600 + idx];
    }
}
