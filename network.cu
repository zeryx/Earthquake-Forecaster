#include "network.h"
#include "getsys.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <fstream>
#include <sstream>
#include <ostream>
#include <utility>
#include <vector>
#include <ctime>
#include <cstdio>
#include <assert.h>

//macros
//cuda error message handling
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); cudaDeviceReset(); exit(EXIT_FAILURE);}} while (0)
#endif
//neural functions
__host__ __device__ inline double sind(double x)
{
    return asin(x * M_PI / 180);
}

__host__ __device__ inline double cosd(double x)
{
    return acos(x * M_PI / 180);
}
__host__ __device__ inline double distCalc(double lat1, double lon1, double lat2, double lon2)
{
    double earthRad = 6371.01;
    double deltalon = abs(lon1 - lon2);
    if(deltalon > 180)
        deltalon = 360 - deltalon;
    double ret = earthRad * atan2( sqrt( pow( cosd(lat1) * sind(deltalon), 2) +
                                         pow( cosd(lat2) * sind(lat1) - sind(lat2) * cosd(lat1) * cosd(deltalon), 2) ),
                                   sind(lat2) * sind(lat1) + cosd(lat2) * cosd(lat1) * cosd(deltalon));
    return ret;
}

__host__ __device__ inline double bearingCalc(double lat1, double lon1, double lat2, double lon2)
{
    double dLon = (lon2 - lon1);

    double y = asin(dLon) * acos(lat2);
    double x = acos(lat1) * asin(lat2) - asin(lat1) * acos(lat2) * acos(dLon);

    double brng = atan2(y, x);

    brng = brng*M_PI/180;
    brng += 360;
    while(brng>= 360)
        brng -= 360;
    brng = 360 - brng;

    return brng;
}

__host__ __device__ inline double ActFunc(double x)
{
    double ret = 1+1/exp(-x);
    return ret;
}
__host__ __device__ inline double normalize(double x, double mean, double stdev)
{
    return (fabs(x-mean))/(stdev*2);;
}

__host__ __device__ inline double shift(double x, double max, double min)
{
    return (x-min)/(max-min);;
}

__global__ void genWeights( kernelArray<double> ref, uint32_t in, kernelArray<int> params, size_t offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = idx*params.array[7]+offset;
    thrust::minstd_rand0 randEng;
    thrust::uniform_real_distribution<double> uniDist(0,1);
    for(int i=0; i<params.array[2]; i++){
        randEng.discard(in+ind);
        ref.array[ind+i] = uniDist(randEng);
    }
    for(int i=params.array[2]; i<params.array[7]; i++){
        ref.array[ind+i]=0;
    }
}

__global__ void Net(kernelArray<double> weights, kernelArray<int> params,
                    kernelArray<double> globalQuakes, kernelArray<int> inputVal, kernelArray<double> siteData,
                    kernelArray<double> answers, kernelArray<std::pair<int, int> > connections,
                    double Kp, int sampleRate,int numOfSites, int hour,
                    double meanCh1, double meanCh2, double meanCh3, double stdCh1, double stdCh2, double stdCh3, size_t offset)
{
    extern __shared__ double scratch[];
    double *When = &scratch[numOfSites*threadIdx.x];
    double *HowCertain = &scratch[numOfSites*blockDim.x + numOfSites*threadIdx.x];
    double *CommunityMag = &scratch[numOfSites*blockDim.x*2 + numOfSites*threadIdx.x];
    for(int i=0; i<numOfSites; i++){
        When[i]=0;
        HowCertain[i]=0;
        CommunityMag[i]=1;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    int ind = idx*params.array[7]+offset;
    typedef std::pair<int, int>*  connectPairMatrix;

    int startOfInput = ind + params.array[2];
    int startOfHidden = startOfInput + params.array[3];
    int startOfMem = startOfHidden + params.array[4];
    int startOfMemGateIn = startOfMem + params.array[5];
    int startOfMemGateOut = startOfMemGateIn + params.array[5];
    int startOfMemGateForget = startOfMemGateOut + params.array[5];
    int startOfOutput = startOfMemGateForget + params.array[5];
    //the weights array carries the neuron scratch space used for the net kernel, I'd like to replace this and reduce the memory allocation asap.
    double *input = &weights.array[startOfInput]; // number of inputs is 9.
    double *hidden = &weights.array[startOfHidden]; // for practice sake, lets say each input has its own neuron (might be true!)
    double *mem = &weights.array[startOfMem]; // stores the input if gate is high
    double *memGateIn = &weights.array[startOfMemGateIn]; //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
    double *memGateOut = &weights.array[startOfMemGateOut];
    double *memGateForget = &weights.array[startOfMemGateForget];
    double *outputs = &weights.array[startOfOutput];
    for(int step=0; step<3600*sampleRate; step++){

        double CommunityLat = 0;
        double CommunityLon = 0;
        for(int j=0; j<sampleRate; j++){//sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
            CommunityLat += siteData.array[j*2]*CommunityMag[j];
            CommunityLon += siteData.array[j*2+1]*CommunityMag[j];
        }
        CommunityLat = CommunityLat/numOfSites;
        CommunityLon = CommunityLon/numOfSites;

        for(int j=0; j<numOfSites; j++){ //each site is run independently of others, but shares an output from the previous step
            double latSite = siteData.array[j*2];
            double lonSite = siteData.array[j*2+1];
            double avgLatGQuake = globalQuakes.array[0];
            double avgLonGQuake = globalQuakes.array[1];
            double GQuakeAvgMag = globalQuakes.array[3];
            double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                        1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0; // n is the weight number
            //            input[0] = normalize(inputVal.array[(3600*sampleRate*j*3 + 0*(3600*sampleRate)+step)], meanCh1, stdCh1);//channel 1
            //            input[1] = normalize(inputVal.array[(3600*sampleRate*j*3 + 2*(3600*sampleRate)+step)], meanCh2, stdCh2);//channel 2
            //            input[2] = normalize(inputVal.array[(3600*sampleRate*j*3 + 3*(3600*sampleRate)+step)], meanCh3, stdCh3);//channel 3
            //            input[3] = shift(GQuakeAvgdist, 40075.1, 0);
            //            input[4] = shift(GQuakeAvgBearing, 360, 0);
            //            input[5] = shift(GQuakeAvgMag, 9.5, 0);
            //            input[6] = shift(Kp, 10, 0);
            //            input[7] = shift(CommunityDist,40075.1/2, 0);
            //            input[8] = shift(CommunityBearing, 360, 0);
            //            //lets reset all neuron values for this new timestep (except memory neurons)
            //            for(int gate=0; gate<params.array[5]; gate++){
            //                memGateIn[gate] = 0;
            //                memGateOut[gate] = 0;
            //                memGateForget[gate] = 0;
            //            }
            //            for(int hid=0; hid<params.array[4]; hid++){
            //                hidden[hid] = 0;
            //            }
            //            for(int out=0; out<params.array[6]; out++){
            //                outputs[out] = 0;
            //            }

            //            //now that everything that should be zeroed is zeroed, lets start the network.
            //            //mem gates & LSTM nodes --
            //            for(int gate = 0; gate<params.array[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateIn
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it); // this needs to be created to use the iterator it correctly.
            //                    if(itr.second == gate+startOfMemGateIn && itr.second < startOfHidden){ //for inputs
            //                        memGateIn[gate] += input[itr.first-startOfInput]*weights.array[ind + n++]; // memGateIn vect starts at 0
            //                    }
            //                    else if(itr.second == gate+startOfMemGateIn && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateIn[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateOut
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == gate+startOfMemGateOut && itr.second < startOfHidden){//for inputs
            //                        memGateOut[gate] += input[itr.first-startOfInput]*weights.array[ind + n++];
            //                    }
            //                    else if(itr.second == gate+startOfMemGateOut && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateOut[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for  memGateForget
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == gate+startOfMemGateForget && itr.second < startOfHidden){//for inputs
            //                        memGateForget[gate] += input[itr.first - startOfInput]*weights.array[ind + n++];
            //                    }
            //                    else if(itr.second == gate+startOfMemGateForget && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateForget[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                memGateIn[gate] = ActFunc(memGateIn[gate]);
            //                memGateOut[gate] = ActFunc(memGateOut[gate]);
            //                memGateForget[gate] = ActFunc(memGateForget[gate]);
            //            }
            //            //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
            //            for (int gate = 0; gate<params.array[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
            //                if(memGateIn[gate] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
            //                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                        if(itr.second == gate+startOfMemGateIn && itr.first < gate+startOfHidden){//only pass inputs
            //                            mem[gate] += input[itr.first-startOfInput]; // no weights attached, but the old value stored here is not removed.
            //                        }
            //                    }
            //                }
            //                if(memGateForget[gate] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
            //                    mem[gate] = 0;
            //                }
            //                //if memGateForget fires, then memGateOut will output nothing.
            //                if(memGateOut[gate] > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
            //                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                        if(itr.first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
            //                            hidden[itr.second-startOfHidden] += mem[gate];
            //                        }
            //                    }
            //                }
            //            }

            //            // hidden neuron nodes --
            //            for(int hid=0; hid<params.array[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){ // Add the inputs to the hidden neurons
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == hid+startOfHidden && itr.first < startOfHidden && itr.first >= startOfInput){ // if an input connects with this hidden neuron
            //                        hidden[hid] += input[itr.first]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == hid+startOfHidden && itr.first < startOfMem && itr.first >= startOfHidden){
            //                        hidden[hid] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                hidden[hid] += 1*weights.array[ind + n++]; // add bias
            //                hidden[hid] = ActFunc(hidden[hid]); // then squash itr.
            //            }
            //            //output nodes --

            //            for(int out =0; out<params.array[6]; out++){// add hidden neurons to the output nodes
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == out+startOfOutput){
            //                        outputs[out] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                outputs[out] += 1*weights.array[ind + n++]; // add bias
            //                outputs[out] = ActFunc(outputs[out]);// then squash itr.
            //            }

            //            When[j*threadIdx.x] += outputs[0]*((2160-hour)-hour)+2160-hour; // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            //            HowCertain[j*threadIdx.x] += outputs[1];
            //            CommunityMag[j*threadIdx.x] =  outputs[2]; // set the next sets communityMag = output #3.
        }
    }
    //    for(int j=0; j<numOfSites; j++){ // now lets get the average when and howcertain values.
    //        When[j*threadIdx.x] = When[j*threadIdx.x]/3600*sampleRate;
    //        HowCertain[j*threadIdx.x] = HowCertain[j*threadIdx.x]/3600*sampleRate;
    //    }
    /*calculate performance for this individual - score = 1/(abs(whenGuess-whenReal)*distToQuake), for whenGuess = when[j] where HowCertain is max for set.
    distToQuake is from the current sites parameters, it emphasizes higher scores for the closest site, a smaller distance is a higher score. */
    //    int maxCertainty=0;
    //    double whenGuess=0;
    //    double latSite=0;
    //    double lonSite=0;
    //    for(int j=0; j<numOfSites; j++){
    //        if(HowCertain[j*threadIdx.x] > maxCertainty){
    //            whenGuess = When[j*threadIdx.x];
    //            latSite = siteData.array[j*2];
    //            lonSite = siteData.array[j*2+1];
    //        }
    //    }
    //    double SiteToQuakeDist = distCalc(latSite, lonSite, answers.array[2], answers.array[3]); // [2] is latitude, [3] is longitude.
    //    double fitness = 1/(abs(whenGuess - answers.array[1]-hour)*SiteToQuakeDist);//larger is better, negative numbers are impossible.
    //    weights.array[ind + params.array[7]-1] = fitness; // set the fitness number for the individual.
}

__global__ void reduce_by_block(kernelArray<double> weights,
                                kernelArray<double> per_block_results,
                                kernelArray<int> params, int n, int device_offset, int blockOffset)
{
    extern __shared__ float sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = idx*params.array[7] + device_offset;

    // load input into __shared__ memory
    float x = 0;
    if(idx < n)
    {
        x = weights.array[ind + params.array[2]];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    // contiguous range pattern
    for(int offset = blockDim.x / 2;
        offset > 0;
        offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            // add a partial sum upstream to our own
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        // wait until all threads in the block have
        // updated their partial sums
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0)
    {
        per_block_results.array[blockIdx.x+blockOffset] = sdata[0];
    }
}

__global__ void swapMemory(kernelArray<double> host, kernelArray<double>device, kernelArray<int>params, size_t host_offset, size_t device_offset){//swap device and host memory in place.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind_dev = idx*params.array[7]+device_offset;
    int ind_host = idx*params.array[7]+host_offset;
    for(int i=0; i<params.array[7]; i++){
        double tmp = device.array[ind_dev+i];
        device.array[ind_dev+i] = host.array[ind_host+i];
        host.array[ind_host+i] = tmp;
    }
}

NetworkGenetic::NetworkGenetic(const int &numInputNodes, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numWeights, std::vector< std::pair<int, int> >&connections){

    _hostParams.array = new int[15];
    _hostParams.size=15;
    _hostParams.array[1] = numInputNodes + numHiddenNeurons + numMemoryNeurons*4 + numOutNeurons; //memory neurons each ahve a rmemeber, forget, and push forward gate neuron.
    _hostParams.array[2] = numWeights;
    _hostParams.array[3] = numInputNodes;
    _hostParams.array[4] = numHiddenNeurons;
    _hostParams.array[5] = numMemoryNeurons;
    _hostParams.array[6] = numOutNeurons;
    _hostParams.array[7] = _hostParams.array[2] + _hostParams.array[1] + 1 + 1; // plus 1 for fitness, plus 1 for community output composite vector
    _connect = &connections;
}

void NetworkGenetic::generateWeights(){
    int blockSize = 512; // number of blocks in the grid
    int gridSize=(_streamSize/_hostParams.array[7])/blockSize;
    size_t global_offset=0;
    size_t device_offset=0;
    for(int n=0; n<_numOfStreams-4; n++){//fill the host first.
        if(n%4==0 && n !=0)
            device_offset =0;
        std::cerr<<"declaring seed..."<<std::endl;
        uint32_t seed;
        FILE *fp;
        fp = std::fopen("/dev/urandom", "r");
        std::fread(&seed, 4, 1, fp);
        std::fclose(fp);
        std::cerr<<"stream number #"<<n<<std::endl;
        std::cerr<<"seed is:"<<seed<<std::endl;
        std::cerr<<"global offset: "<<global_offset<<std::endl;
        std::cerr<<"device offset: "<<device_offset<<std::endl<<std::endl;
        genWeights<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, _deviceParams, device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[global_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
        global_offset += _streamSize;
        device_offset += _streamSize;
    }
    device_offset=0;
    for(int n=_numOfStreams-4; n<_numOfStreams; n++){//fill the gpu last
        uint32_t seed;
        FILE *fp;
        fp = std::fopen("/dev/urandom", "r");
        std::fread(&seed, 4, 1, fp);
        std::fclose(fp);
        std::cerr<<"stream number #"<<n<<std::endl;
        std::cerr<<"seed is:"<<seed<<std::endl;
        std::cerr<<"device offset: "<<device_offset<<std::endl<<std::endl;
        genWeights<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, _deviceParams, device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        device_offset += _streamSize;
    }
    //    for(int i=0; i<(10000); i++){
    //        if(i%83==0)
    //            std::cerr<<"for this individual:"<<std::endl;
    //        std::cerr<<i<<" is: "<<host_genetics.array[i]<<std::endl;
    //    }
    //    std::cerr<<"the size of an individual is: "<<_hostParams.array[7]<<std::endl;
    //    std::cerr<<"stream size is:"<<_streamSize<<std::endl;
}


void NetworkGenetic::allocateHostAndGPUObjects( float pMax, size_t deviceRam, size_t hostRam){
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
    std::cerr<<"RIGHT BEFORE DEVICE"<<std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_deviceParams.array, _hostParams.size*sizeof(int)));
    _deviceParams.size = _hostParams.size;
    CUDA_SAFE_CALL(cudaMemcpy(&_deviceParams.array[0], &_hostParams.array[0], _hostParams.size*sizeof(int), cudaMemcpyHostToDevice));
    size_t totalHost = hostRam*pMax;
    size_t totalDevice = deviceRam*pMax;
    std::cerr<<"total free device ram : "<<deviceRam<<std::endl;
    std::cerr<<"total free host ram : "<<hostRam<<std::endl;
    while(totalHost%_hostParams.array[7] || totalHost%sizeof(double) || totalHost%(sizeof(double)*4) || totalHost%512) // get the largest number divisible by the individual size, the threads in a block, and the size of a double
        totalHost= totalHost -1;
    while(totalDevice%_hostParams.array[7] || totalDevice%sizeof(double) || totalDevice%(sizeof(double)*4) || totalDevice%512)
        totalDevice = totalDevice -1;
    //make each of the memory arguments divisible by 512 (threads per block)
    _streamSize = totalDevice/(sizeof(double)*4);
    _streambytes = _streamSize*sizeof(double);
    std::cerr<<"bytes per stream :"<<_streambytes<<std::endl;
    assert(_streambytes == _streamSize*sizeof(double));
    _numOfStreams = ceil((totalDevice/sizeof(double) +totalHost/sizeof(double))/_streamSize); // number of streams = total array alloc / number of streams.
    assert(_streambytes * _numOfStreams <= totalDevice+totalHost);
    std::cerr<<"number of streams: "<<_numOfStreams<<std::endl;
    device_genetics.size = (totalDevice)/sizeof(double);
    host_genetics.size = totalHost/sizeof(double);
    std::cerr<<"device ram to allocate: "<<totalDevice<<std::endl;
    std::cerr<<"host ram to allocate: "<<totalHost<<std::endl;
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&host_genetics.array, totalHost, cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)&host_genetics_device.array, (void*)host_genetics.array, 0 ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &device_genetics.array, totalDevice));
    std::cerr<<"all allocated, moving on."<<std::endl;
    _stream.resize(_numOfStreams);
    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream.at(i)));
        CUDA_SAFE_CALL( cudaStreamQuery(_stream.at(i)));
    }
}
bool NetworkGenetic::init(int sampleRate, int SiteNum, std::vector<double> *siteData){
    _sampleRate = sampleRate;
    _numofSites = SiteNum;
    _siteData = siteData;
    _istraining = false;
    return true;
}

bool NetworkGenetic::checkForWeights(std::string filepath){
    std::ifstream weightFile(filepath.c_str(), std::ios_base::ate | std::ios_base::binary);
    std::cerr<<"checking for weights.."<<std::endl;
    if(weightFile){
        std::cerr<<"the weightfile exists"<<std::endl;
        std::string line;
        int filesize = weightFile.tellg();
        weightFile.seekg(0, weightFile.beg);
        int itr =0;
        this->allocateHostAndGPUObjects(0.85, GetDeviceRamInBytes(), filesize - GetDeviceRamInBytes());
        for( int n=0; n<_numOfStreams; n++){
            int offset = n*_streambytes/sizeof(double);
            CUDA_SAFE_CALL(cudaMemset(&device_genetics.array, 0, _streambytes));
            while(std::getline(weightFile, line) && itr <= device_genetics.size){ // each line
                std::string item;
                std::stringstream ss(line);
                while(std::getline(ss, item, ',') && itr <= device_genetics.size){ // each weight
                    device_genetics.array[itr] = std::atoi(item.c_str());
                }
            }
            CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[offset], &device_genetics.array[offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
            itr = 0;
        }
        weightFile.close();
        return true;
    }
    else{
        std::cerr<<"no weightfile found"<<std::endl;
        weightFile.close();
        return false;
    }
}

void NetworkGenetic::doingTraining(int site, int hour, double lat,
                                   double lon, double mag, double dist){
    _answers.push_back(site);
    _answers.push_back(hour);
    _answers.push_back(lat);
    _answers.push_back(lon);
    _answers.push_back(mag);
    _answers.push_back(dist);
    _istraining = true;
}

void NetworkGenetic::storeWeights(std::string filepath){
    std::ofstream ret;
    ret.open(filepath.c_str(), std::ios_base::out | std::ios_base::trunc);
    for(int i=0; i<device_genetics.size; i++){
        ret << device_genetics.array[i]<<","<<std::endl;
    }
    ret.close();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamDestroy(_stream[i]));
    }
    CUDA_SAFE_CALL(cudaFree(device_genetics.array));
    CUDA_SAFE_CALL(cudaFree(host_genetics.array));
    CUDA_SAFE_CALL(cudaDeviceReset());
}

void NetworkGenetic::forecast(std::vector<double> *ret, int &hour, std::vector<int> *data, double &Kp, std::vector<double> *globalQuakes)
{
    //were going to normalize the inputs using v` = v-mean/stdev, so we need mean and stdev for each channel.
    double meanCh1=0, meanCh2=0, meanCh3=0, stdCh1=0, stdCh2=0, stdCh3=0;
    int num=0;
    std::cerr<<"right before mean & std calc"<<std::endl;
    for(int i=0; i<3600*_sampleRate; i++){
        for(int j=0; j < _numofSites; j++){
            meanCh1 += data->at(3600*_sampleRate*j*3 + 0*3600*_sampleRate+i);
            meanCh2 += data->at(3600*_sampleRate*j*3 + 1*3600*_sampleRate+i);
            meanCh3 += data->at(3600*_sampleRate*j*3 + 2*3600*_sampleRate+i);
            num++;
        }
    }
    meanCh1 = meanCh1/num;
    meanCh2 = meanCh2/num;
    meanCh3 = meanCh3/num;
    stdCh1 = sqrt(meanCh1);
    stdCh2 = sqrt(meanCh2);
    stdCh3 = sqrt(meanCh3);
    std::cerr<<"means are: "<<meanCh1<<" "<<meanCh2<<" "<<meanCh3<<std::endl;
    std::cerr<<"stdevs are: "<<stdCh1<<" "<<stdCh2<<" "<<stdCh3<<std::endl;
    std::cerr<<"channels std and mean calculated"<<std::endl;
    //input data from all sites and all channels normalized
    if(_istraining == true){
        std::cerr<<"beginning of training"<<std::endl;
        kernelArray<double>retVec, gQuakeAvg, answers, siteData, partial_reduce_sums;
        kernelArray<int> input;
        kernelArray<std::pair<int, int> > dConnect;
        int netBlockSize = 64; // the actual grid size needed
        int regBlockSize = 512;
        size_t reduceGridSize = (_streamSize/_hostParams.array[7])/regBlockSize + (((_streamSize/_hostParams.array[7])%regBlockSize) ? 1 : 0);
        size_t regGridSize = (_streamSize/_hostParams.array[7])/regBlockSize;
        size_t netGridSize = (_streamSize/_hostParams.array[7])/netBlockSize;
        CUDA_SAFE_CALL(cudaMalloc((void**)&input.array, data->size()*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&retVec.array, ret->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&gQuakeAvg.array, globalQuakes->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&answers.array, _answers.size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dConnect.array, _connect->size()*sizeof(std::pair<int, int>)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&siteData.array, _siteData->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&partial_reduce_sums.array, _numOfStreams*(regBlockSize+1)*sizeof(double)));
        input.size = data->size();
        retVec.size = 2160*_numofSites;
        gQuakeAvg.size = globalQuakes->size();
        answers.size = _answers.size();
        dConnect.size = _connect->size();
        siteData.size = _siteData->size();
        partial_reduce_sums.size = _numOfStreams*(regBlockSize+1);
        CUDA_SAFE_CALL(cudaMemcpyAsync(input.array, data->data(), input.size, cudaMemcpyHostToDevice, _stream[0]));
        CUDA_SAFE_CALL(cudaMemcpyAsync(gQuakeAvg.array, globalQuakes->data(), gQuakeAvg.size, cudaMemcpyHostToDevice, _stream[1]));
        CUDA_SAFE_CALL(cudaMemcpyAsync(dConnect.array, _connect->data(), dConnect.size, cudaMemcpyHostToDevice, _stream[2]));
        CUDA_SAFE_CALL(cudaMemcpyAsync(siteData.array, _siteData->data(), siteData.size, cudaMemcpyHostToDevice, _stream[3]));
        CUDA_SAFE_CALL(cudaMemcpyAsync(answers.array, _answers.data(), answers.size, cudaMemcpyHostToDevice, _stream[4]));
        CUDA_SAFE_CALL(cudaMemsetAsync(retVec.array, 0, retVec.size*sizeof(double), _stream[5]));
        CUDA_SAFE_CALL(cudaMemsetAsync(partial_reduce_sums.array, 0, partial_reduce_sums.size*sizeof(double), _stream[6]));
        std::cerr<<"after copy of training"<<std::endl;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double fitnessAvg=0;
        int fitItr=0;
        size_t host_offset = 0;
        size_t device_offset=0;
        for(int n=0; n<_numOfStreams-4; n++){
            if(n%4==0 && n!=0)
                device_offset=0;
            std::cerr<<"stream number: "<<n<<std::endl;
            std::cerr<<"host offset: "<<host_offset<<std::endl;
            std::cerr<<"device offset: "<<device_offset<<std::endl;
            Net<<<netGridSize, netBlockSize, netBlockSize*sizeof(double)*3*_numofSites, _stream[n]>>>(device_genetics, _deviceParams, gQuakeAvg,
                                                                                                      input, siteData, answers,
                                                                                                      dConnect,Kp,_sampleRate,_numofSites, hour,
                                                                                                      meanCh1, meanCh2, meanCh3, stdCh1, stdCh2, stdCh3, device_offset);
            CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::cerr<<"net completed."<<std::endl;
            //            std::cerr<<"reduce grid size: "<<reduceGridSize<<std::endl;
            //            reduce_by_block<<<reduceGridSize, blockSize, blockSize*sizeof(double), _stream[n]>>>(device_genetics,
            //                                                                                                 partial_reduce_sums,
            //                                                                                                 _deviceParams, _streamSize/_deviceParams.array[7], device_offset, reduceGridSize*n);
            //            CUDA_SAFE_CALL(cudaPeekAtLastError());
            std::cerr<<"swap grid size:"<<regGridSize<<std::endl;
            swapMemory<<<regGridSize, regBlockSize, 0, _stream[n]>>>(host_genetics_device, device_genetics, _deviceParams, host_offset, device_offset);
            CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            std::cerr<<"memory swap completed"<<std::endl;
            //            for(int itr =0; itr<partial_reduce_sums.size; itr++){
            //                fitnessAvg += partial_reduce_sums.array[itr];
            //                fitItr++;
            //            }
            host_offset += _streamSize;
            device_offset += _streamSize;
        }
        device_offset=0;
        for(int n=_numOfStreams-4; n<_numOfStreams; n++){
            std::cerr<<"stream number: "<<n<<std::endl;
            std::cerr<<"host offset: "<<host_offset<<std::endl;
            std::cerr<<"device offset: "<<device_offset<<std::endl;
            Net<<<regGridSize, regBlockSize, regBlockSize*sizeof(double)*3*_numofSites, _stream[n]>>>(device_genetics, _deviceParams, gQuakeAvg,
                                                                                                      input, siteData, answers,
                                                                                                      dConnect,Kp,_sampleRate,_numofSites, hour,
                                                                                                      meanCh1, meanCh2, meanCh3, stdCh1, stdCh2, stdCh3, device_offset);
            device_offset += _streamSize;
        }
        //        fitnessAvg = fitnessAvg /fitItr;
        //        std::cerr<<"the average fitness for this round is: "<<fitnessAvg<<std::endl;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaFree(dConnect.array));
        CUDA_SAFE_CALL(cudaFree(input.array));
        CUDA_SAFE_CALL(cudaFree(gQuakeAvg.array));
        CUDA_SAFE_CALL(cudaFree(retVec.array));
        CUDA_SAFE_CALL(cudaFree(answers.array));
        CUDA_SAFE_CALL(cudaFree(siteData.array));
        CUDA_SAFE_CALL(cudaFree(partial_reduce_sums.array));
    }
    else{
        std::cerr<<"entered not training version.."<<std::endl;
        typedef std::vector<std::pair<int, int> > connectPairMatrix;
        //replace this later
        _best.resize(_hostParams.array[2]);
        for(std::vector<double>::iterator it = _best.begin(); it != _best.end(); ++it){
            std::srand(std::time(NULL)+*it);
            *it = (double)(std::rand())/(RAND_MAX);
        }
        std::cerr<<"example best vector has been set."<<std::endl;
        double CommunityLat = 0;
        double CommunityLon = 0;
        std::vector<double> When(_numofSites, 0);
        std::vector<double> HowCertain(_numofSites,0);
        std::vector<double> CommunityMag(_numofSites, 1); //give all sites equal mag to start, this value is [0,1]
        std::cerr<<"all output vectors created and initialized."<<std::endl;
        for(int step=0; step<3600*_sampleRate; step++){
            std::cerr<<"entering step #"<<step<<std::endl;
            for(int j=0; j<_numofSites; j++){ //sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
                CommunityLat += _siteData->at(j*2)*CommunityMag[j];
                CommunityLon += _siteData->at(j*2+1)*CommunityMag[j];
            }
            CommunityLat = CommunityLat/_numofSites;
            CommunityLon = CommunityLon/_numofSites;

            for(int j=0; j<_numofSites; j++){ // each site is run independently of others, but shares an output from the previous step
                std::cerr<<"entering site #"<<j<<std::endl;
                double latSite = _siteData->at(j*2);
                double lonSite = _siteData->at(j*2+1);
                double avgLatGQuake = globalQuakes->at(0);
                double avgLonGQuake = globalQuakes->at(1);
                double GQuakeAvgMag = globalQuakes->at(3);
                double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
                double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
                std::vector<double> input;
                std::vector<double> hidden, outputs, mem, memGateOut, memGateIn, memGateForget;
                //replace these with real connections, num of inputs, and num of hidden & memory neurons (mem neurons probably accurate)
                input.resize(_hostParams.array[3], 0); // number of inputs is 9.
                hidden.resize(_hostParams.array[4], 0); // for practice sake, lets say each input has its own neuron (might be true!)
                mem.resize(_hostParams.array[5], 0); // stores the input if gate is high
                memGateOut.resize(_hostParams.array[5], 0); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
                memGateIn.resize(_hostParams.array[5], 0);
                memGateForget.resize(_hostParams.array[5], 0);
                outputs.resize(_hostParams.array[6], 0); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                    1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
                std::cerr<<"all neuron vectors are sized, all pre-net calculations done."<<std::endl;
                int n =0;
                int startOfInput = 0;
                int startOfHidden = startOfInput +_hostParams.array[3];
                int startOfMem = startOfHidden + _hostParams.array[4];
                int startOfMemGateIn = startOfMem + _hostParams.array[5];
                int startOfMemGateOut = startOfMemGateIn + _hostParams.array[5];
                int startOfMemGateForget = startOfMemGateOut + _hostParams.array[5];
                int startOfOutput = startOfMemGateForget + _hostParams.array[5];
                input[0] = normalize((double)(data->at(3600*_sampleRate*j*3 + 0*(3600*_sampleRate)+step)), meanCh1, stdCh1);
                input[1] = normalize((double)(data->at(3600*_sampleRate*j*3 + 1*(3600*_sampleRate)+step)), meanCh2, stdCh2);
                input[2] = normalize((double)(data->at(3600*_sampleRate*j*3 + 2*(3600*_sampleRate)+step)), meanCh3, stdCh3);
                input[3] = shift(GQuakeAvgdist, 40075.1, 0);
                input[4] = shift(GQuakeAvgBearing, 360, 0);
                input[5] = shift(GQuakeAvgMag, 9.5, 0);
                input[6] = shift(Kp, 10, 0);
                input[7] = shift(CommunityDist,40075.1/2, 0);
                input[8] = shift(CommunityBearing, 360, 0);
                //lets reset all neuron values for this new timestep (except memory neurons)
                for(int gate=0; gate<_hostParams.array[5]; gate++){
                    memGateIn.at(gate) = 0;
                    memGateOut.at(gate) = 0;
                    memGateForget.at(gate) = 0;
                }
                for(int hid=0; hid<_hostParams.array[4]; hid++){
                    hidden[hid] = 0;
                }
                for(int out=0; out<_hostParams.array[6]; out++){
                    outputs[out] = 0;
                }
                std::cerr<<"memGate, hidden, and output neurons are zeroed."<<std::endl;
                //now that everything that should be zeroed is zeroed, lets start the network.
                //mem gates & LSTM nodes --
                std::cerr<<"preparing to set the values for memoryGates."<<std::endl;
                for(int gate = 0; gate<_hostParams.array[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateIn
                        if(it->second == gate+startOfMemGateIn && it->first < startOfHidden){ //for inputs
                            std::cerr<<"weights for memGateIn #"<<gate<<" is: "<<_best[n];
                            memGateIn.at(gate) += input[it->first-startOfInput]*_best[n++]; // memGateIn vect starts at 0
                        }
                        else if(it->second == gate+startOfMemGateIn && it->first >startOfHidden && it->first < startOfMem){//for hidden neurons
                            memGateIn.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateOut
                        if(it->second == gate+startOfMemGateOut && it->first < startOfHidden){//for inputs
                            std::cerr<<"weights for memGateOut #"<<gate<<" is: "<<_best[n];
                            memGateOut.at(gate) += input[it->first-startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateOut && it->first >startOfHidden && it->first <startOfMem){//for hidden neurons
                            memGateOut.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for  memGateForget
                        if(it->second == gate+startOfMemGateForget && it->first < startOfHidden){//for inputs
                            std::cerr<<"weights for memGateForget #"<<gate<<" is: "<<_best[n];
                            memGateForget.at(gate) += input[it->first - startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateForget && it->first >startOfHidden && it->first <startOfMem){//for hidden neurons
                            memGateForget.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    memGateIn.at(gate) = ActFunc(memGateIn.at(gate));
                    memGateOut.at(gate) = ActFunc(memGateOut.at(gate));
                    memGateForget.at(gate) = ActFunc(memGateForget.at(gate));
                    std::cerr<<"memGateIn val: "<<memGateIn.at(gate)<<std::endl;
                    std::cerr<<"memGateOut val: "<<memGateOut.at(gate)<<std::endl;
                    std::cerr<<"memGateForget val: "<<memGateForget.at(gate)<<std::endl;
                }
                //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
                for (int gate = 0; gate<_hostParams.array[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
                    if(memGateIn.at(gate) > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->second == gate+startOfMemGateIn && it->first < gate+startOfHidden){//only pass inputs
                                mem.at(gate) += input[it->first-startOfInput]; // no weights attached, but the old value stored here is not removed.
                            }
                        }
                    }
                    if(memGateForget.at(gate) > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
                        mem.at(gate) = 0;
                    }
                    //if memGateForget fires, then memGateOut will output nothing.
                    if(memGateOut.at(gate) > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
                                hidden[it->second-startOfHidden] += mem.at(gate);
                            }
                        }
                    }
                    std::cerr<<"mem val stored is: "<<mem.at(gate)<<std::endl;
                }

                // hidden neuron nodes --
                for(int hid=0; hid<_hostParams.array[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){ // Add the inputs to the hidden neurons
                        if(it->second == hid+startOfHidden && it->first < startOfHidden){ // if an input connects with this hidden neuron
                            hidden[hid] += input[it->first]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
                        if(it->second == hid+startOfHidden && it->first < startOfMem && it->first > startOfHidden){
                            hidden[hid] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    hidden[hid] += 1*_best[n++]; // add bias
                    hidden[hid] = ActFunc(hidden[hid]); // then squash it.
                    std::cerr<<"hidden nueron values: "<<hidden[hid]<<std::endl;
                }
                //output nodes --

                for(int out =0; out<_hostParams.array[6]; out++){// add hidden neurons to the output nodes
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                        if(it->second == out+startOfOutput){
                            outputs[out] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    outputs[out] += 1*_best[n++]; // add bias
                    outputs[out] = ActFunc(outputs[out]);// then squash it.
                }


                When[j] += outputs[0]*((2160-hour)-hour)+2160-hour; //return when back to an integer value (adjust to fit within boundaries)
                std::cerr<<"When for site: "<<j<<" and for step: "<<step<< " is: "<<When[j]<<std::endl;
                HowCertain[j] += outputs[1];
                std::cerr<<"howCertain for site: "<<j<<" and for step: "<<step<< " is: "<<HowCertain[j]<<std::endl;
                CommunityMag[j] =  outputs[2]; // set the next sets communityMag = output #3.
                std::cerr<<"ComunityMagnitude for site: "<<j<<" and for step: "<<step<< " is: "<<CommunityMag[j]<<std::endl;
            }
        }
        for(int j=0; j<_numofSites; j++){ // each site has its own when and howcertain vector
            When[j] = When[j]/3600*_sampleRate;
            HowCertain[j] = HowCertain[j]/3600*_sampleRate;
        }
        //all done, lets output the return matrix.
        //since right now were using a point value for when & how certain (only one output per site),
        //we're going to approximate using a normal distribution around when with a sigma of howCertain, over the whole array from T=currentHour [T, 2160]
        for(int h=hour; h<2160; h++){
            for(int j=0; j<_numofSites; j++){
                ret->at(h*_numofSites+j)= 1/(1/HowCertain[j]*sqrt(2*M_PI))*exp(-pow(h-When[j], 2)/(2*pow(1/HowCertain[j], 2))); // normal distribution with a mu of When and a sigma of 1/HowCertain
            }
        }
    }
}
