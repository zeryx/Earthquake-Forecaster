#include "network.h"
#include "getsys.h"
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <sstream>
#include <ostream>
#include <utility>
#include <vector>
#include <ctime>
#include <assert.h>

//macros
//cuda error message handling
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); exit(EXIT_FAILURE);}} while (0)
#endif
//neural functions
__host__ __device__ inline double sind(double x)
{
    double ret= sin(x * M_PI / 180);
    return ret;
}

__host__ __device__ inline double cosd(double x)
{
    return cos(x * M_PI / 180);
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

    double y = sin(dLon) * cos(lat2);
    double x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon);

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
    double ret = (abs(x-mean))/stdev*2;
    return ret;
}

__host__ __device__ inline double shift(double x, double max, double min)
{
    double ret = (x-min)/(max-min);
    return ret;
}

__global__ void genWeights( unifiedArray<double> ref, long in, dataArray<int> params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ind = idx*params.array[7];
    thrust::minstd_rand0 randEng;
    randEng.seed(idx);
    int seed = idx+ref.size*in;
    thrust::uniform_real_distribution<double> uniDist(0,1);
    for(int i=0; i<params.array[2]; i++){
        randEng.discard(seed+1);
        ref.array[ind+i] = uniDist(randEng);
    }
}

__global__ void Net(unifiedArray<double> weights, dataArray<int> params,
                    dataArray<double> globalQuakes, dataArray<int> inputVal, dataArray<double> siteData,
                    dataArray<double> answers, dataArray<thrust::pair<int, int> > connections,
                    double Kp, int sampleRate,int numOfSites, int hour,
                    double meanCh1, double meanCh2, double meanCh3, double stdCh1, double stdCh2, double stdCh3)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    int ind = idx*params.array[7];
    typedef thrust::device_ptr<thrust::pair<int, int> >  connectPairMatrix;
    double CommunityLat = 0;
    double CommunityLon = 0;
    double *When = new double[numOfSites];
    double *HowCertain = new double[numOfSites];
    double *CommunityMag = new double[numOfSites]; //give all sites equal mag to start, this value is [0,1]

    for(int step=0; step<3600*sampleRate; step++){

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
            //double avgDepthGQuake = globalQuakes.array[2); don't think I care about depth that much.
            double GQuakeAvgMag = globalQuakes.array[3];
            double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0;
            int startOfInput = 0;
            int startOfHidden = startOfInput +params.array[3];
            int startOfMem = startOfHidden + params.array[4];
            int startOfMemGateIn = startOfMem + params.array[5];
            int startOfMemGateOut = startOfMemGateIn + params.array[5];
            int startOfMemGateForget = startOfMemGateOut + params.array[5];
            int startOfOutput = startOfMemGateForget + params.array[5];
            // the weights array carries the neuron scratch space used for the net kernel, I'd like to replace this and reduce the memory allocation asap.
            double *input = &weights.array[startOfInput]; // number of inputs is 9.
            double *hidden = &weights.array[startOfHidden]; // for practice sake, lets say each input has its own neuron (might be true!)
            double *mem = &weights.array[startOfMem]; // stores the input if gate is high
            double *memGateIn = &weights.array[startOfMemGateIn]; //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
            double *memGateOut = &weights.array[startOfMemGateOut];
            double *memGateForget = &weights.array[startOfMemGateForget];
            double *outputs = &weights.array[startOfOutput];

            input[0] = normalize(inputVal.array[(3600*sampleRate*j*3 + 1*(3600*sampleRate)+step)], meanCh1, stdCh1);//channel 1
            input[1] = normalize(inputVal.array[(3600*sampleRate*j*3 + 2*(3600*sampleRate)+step)], meanCh2, stdCh2);//channel 2
            input[2] = normalize(inputVal.array[(3600*sampleRate*j*3 + 3*(3600*sampleRate)+step)], meanCh3, stdCh3);//channel 3
            input[3] = shift(GQuakeAvgdist, 40075.1, 0);
            input[4] = shift(GQuakeAvgBearing, 360, 0);
            input[5] = shift(GQuakeAvgMag, 9.5, 0);
            input[6] = shift(Kp, 10, 0);
            input[7] = shift(CommunityDist,40075.1/2, 0);
            input[8] = shift(CommunityBearing, 360, 0);
            //lets reset all neuron values for this new timestep (except memory neurons)
            for(int gate=0; gate<params.array[5]; gate++){
                memGateIn[gate] = 0;
                memGateOut[gate] = 0;
                memGateForget[gate] = 0;
            }
            for(int hid=0; hid<params.array[4]; hid++){
                hidden[hid] = 0;
            }
            for(int out=0; out<params.array[6]; out++){
                outputs[out] = 0;
            }

            //now that everything that should be zeroed is zeroed, lets start the network.
            //mem gates & LSTM nodes --
            for(int gate = 0; gate<params.array[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateIn
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it); // this needs to be created to use the iterator it correctly.
                    if(itr.second == gate+startOfMemGateIn && itr.second < startOfHidden){ //for inputs
                        memGateIn[gate] += input[itr.first-startOfInput]*weights.array[ind + n++]; // memGateIn vect starts at 0
                    }
                    else if(itr.second == gate+startOfMemGateIn && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        memGateIn[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateOut
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                    if(itr.second == gate+startOfMemGateOut && itr.second < startOfHidden){//for inputs
                        memGateOut[gate] += input[itr.first-startOfInput]*weights.array[ind + n++];
                    }
                    else if(itr.second == gate+startOfMemGateOut && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        memGateOut[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for  memGateForget
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                    if(itr.second == gate+startOfMemGateForget && itr.second < startOfHidden){//for inputs
                        memGateForget[gate] += input[itr.first - startOfInput]*weights.array[ind + n++];
                    }
                    else if(itr.second == gate+startOfMemGateForget && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        memGateForget[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
                    }
                }
                memGateIn[gate] = ActFunc(memGateIn[gate]);
                memGateOut[gate] = ActFunc(memGateOut[gate]);
                memGateForget[gate] = ActFunc(memGateForget[gate]);
            }
            //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
            for (int gate = 0; gate<params.array[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
                if(memGateIn[gate] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                        thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                        if(itr.second == gate+startOfMemGateIn && itr.first < gate+startOfHidden){//only pass inputs
                            mem[gate] += input[itr.first-startOfInput]; // no weights attached, but the old value stored here is not removed.
                        }
                    }
                }
                if(memGateForget[gate] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
                    mem[gate] = 0;
                }
                //if memGateForget fires, then memGateOut will output nothing.
                if(memGateOut[gate] > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                        thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                        if(itr.first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
                            hidden[itr.second-startOfHidden] += mem[gate];
                        }
                    }
                }
            }

            // hidden neuron nodes --
            for(int hid=0; hid<params.array[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){ // Add the inputs to the hidden neurons
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                    if(itr.second == hid+startOfHidden && itr.first < startOfHidden && itr.first >= startOfInput){ // if an input connects with this hidden neuron
                        hidden[hid] += input[itr.first]*weights.array[ind + n++];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                    if(itr.second == hid+startOfHidden && itr.first < startOfMem && itr.first >= startOfHidden){
                        hidden[hid] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
                    }
                }
                hidden[hid] += 1*weights.array[ind + n++]; // add bias
                hidden[hid] = ActFunc(hidden[hid]); // then squash itr.
            }
            //output nodes --

            for(int out =0; out<params.array[6]; out++){// add hidden neurons to the output nodes
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                    thrust::pair<int, int>itr = static_cast<thrust::pair<int, int> >(*it);
                    if(itr.second == out+startOfOutput){
                        outputs[out] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
                    }
                }
                outputs[out] += 1*weights.array[ind + n++]; // add bias
                outputs[out] = ActFunc(outputs[out]);// then squash itr.
            }

            When[j] += outputs[0]*((2160-hour)-hour)+2160-hour; // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            HowCertain[j] += outputs[1];
            CommunityMag[j] =  outputs[2]; // set the next sets communityMag = output #3.
        }
    }
    for(int j=0; j<numOfSites; j++){ // now lets get the average when and howcertain values.
        When[j] = When[j]/3600*sampleRate;
        HowCertain[j] = HowCertain[j]/3600*sampleRate;
    }
    // calculate performance for this individual - score = 1/(abs(whenGuess-whenReal)*distToQuake), for whenGuess = when[j] where HowCertain is max for set.
    //distToQuake is from the current sites parameters, it emphasizes higher scores for the closest site, a smaller distance is a higher score.
    int maxCertainty=0;
    double whenGuess=0;
    double latSite;
    double lonSite;
    for(int j=0; j<numOfSites; j++){
        if(HowCertain[j] > maxCertainty){
            whenGuess = When[j];
            latSite = siteData.array[j*2];
            lonSite = siteData.array[j*2+1];
        }
    }
    double SiteToQuakeDist = distCalc(latSite, lonSite, answers.array[2], answers.array[3]); // [2] is latitude, [3] is longitude.
    double fitness = 1/(abs(whenGuess - answers.array[1]-hour)*SiteToQuakeDist);//larger is better, negative numbers are impossible.
    weights.array[ind + params.array[2]+2] = fitness; // set the fitness number for the individual.
}

__global__ void reduce_by_block(unifiedArray<double> weights,
                                dataArray<double> per_block_results,
                                dataArray<int> params)
{
    extern __shared__ float sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int fit = idx*params.array[7]+params.array[2]+2;

    // load input into __shared__ memory
    float x = 0;
    if(idx < params.array[8])
    {
        x = weights.array[fit];
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
        per_block_results.array[blockIdx.x] = sdata[0];
    }
}

__global__ void swapMemory(unifiedArray<double> device, unifiedArray<double>host, int offset){//swap device and host memory in place.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double tmp;
    tmp = device.array[idx];
    device.array[idx] = host.array[idx+offset];
    host.array[idx+offset] = tmp;
}

NetworkGenetic::NetworkGenetic(const int &numInputNodes, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numWeights, std::vector< thrust::pair<int, int> >&connections){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInputNodes + numHiddenNeurons + numMemoryNeurons*4 + numOutNeurons; //memory neurons each ahve a rmemeber, forget, and push forward gate neuron.
    _NNParams[2] = numWeights;
    _NNParams[3] = numInputNodes;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = _NNParams[2] + _NNParams[1] + 1 + 1; // plus 1 for fitness, plus 1 for community output composite vector
    _connect = &connections;
}

void NetworkGenetic::initializeWeights(){
    int gridSize; //the blocksize defined by the configurator
    int blockSize = 512; // number of blocks in the grid
    int seedItr = 0;
    int global_offset=0;
    for(int n=0; n<_numOfStreams; n++){
        if(n <_numOfStreams-1){
            long seed = std::clock() + std::clock()*seedItr++;
            gridSize=(_streamSize/_NNParams[7])/(blockSize);
            std::cerr<<"number of blocks is: "<<gridSize<<std::endl;
            std::cerr<<"total num of individuals in this device: "<<gridSize*blockSize<<std::endl;
            std::cerr<<"stream number #"<<n<<std::endl;
            std::cerr<<"seed is:"<<seed<<std::endl;
            std::cerr<<"global offset: "<<global_offset<<std::endl;
            genWeights<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, convertToKernel(_NNParams));
            CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[global_offset], &device_genetics.array[0], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));

            global_offset += _streamSize;
        }
        else{//host ram is full, fill the GPU now and then were done.
            long seed = std::clock() + std::clock()*seedItr++;
            gridSize=(_streambytes/_NNParams[7])/(blockSize); // round down isntead of up.
            std::cerr<<"stream number #"<<n<<std::endl;
            std::cerr<<"seed is:"<<seed<<std::endl;
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            genWeights<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, convertToKernel(_NNParams));
            CUDA_SAFE_CALL(cudaPeekAtLastError());

        }
    }
}


void NetworkGenetic::allocateHostAndGPUObjects( float pMax, size_t deviceRam, size_t hostRam){
    size_t totalHost = hostRam;
    size_t totalWeights = deviceRam; // the number of neurons has precident on the number of devices.
    size_t totalNeurons;
    std::cerr<<"total free device ram : "<<deviceRam<<std::endl;
    std::cerr<<"total free host ram : "<<hostRam<<std::endl;
    totalHost = (totalHost*pMax*_NNParams[7])/(_NNParams[7]);
    totalWeights = (totalWeights*pMax*_NNParams[7])/(_NNParams[7]);
    totalWeights = totalWeights - _NNParams[1]*(totalWeights/_NNParams[7]);
    totalNeurons = (_NNParams[1])*(totalWeights/_NNParams[7]);
    //make each of the memory arguments divisible by 512 (threads per block)
    _streambytes = (totalNeurons+totalWeights);
    _streamSize = _streambytes/sizeof(double);
    assert(_streambytes == (totalNeurons + totalWeights));
    _numOfStreams = ceil((totalNeurons + totalWeights +totalHost)/_streambytes); // number of streams = total array alloc / number of streams.
    assert(_streambytes * _numOfStreams <= totalNeurons+totalWeights+totalHost);
    std::cerr<<"bytes per stream :"<<_streambytes<<std::endl;
    std::cerr<<"number of streams: "<<_numOfStreams<<std::endl;
    device_genetics.size = (totalWeights+totalNeurons)/sizeof(double);
    host_genetics.size = totalHost/sizeof(double);
    std::cerr<<"device ram to allocate: "<<totalWeights<<std::endl;
    std::cerr<<"host ram to allocate: "<<totalHost<<std::endl;
    std::cerr<<"neuron ram to allocate: "<<totalNeurons<<std::endl;
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&host_genetics.array, totalHost, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_SAFE_CALL(cudaHostGetDevicePointer((void**)&host_genetics_device.array, (void*)host_genetics.array, 0 ));
    CUDA_SAFE_CALL(cudaMalloc((void**) &device_genetics.array, totalWeights + totalNeurons));
    std::cerr<<"all allocated, moving on."<<std::endl;
    _stream.resize(_numOfStreams);
    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream.at(i)));
        CUDA_SAFE_CALL( cudaStreamQuery(_stream.at(i)));
    }
}
bool NetworkGenetic::init(int sampleRate, int SiteNum, std::vector<double> siteData){
    _sampleRate = sampleRate;
    _numofSites = SiteNum;
    _siteData.resize(siteData.size());
    try{thrust::copy(siteData.begin(), siteData.end(), _siteData.begin());}
    catch(thrust::system_error &e){
        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
        exit(-1);
    }
    catch(std::bad_alloc &e){
        std::cerr<<"Ran out of space due to : "<<"host"<<std::endl;
        std::cerr<<e.what()<<std::endl;
        exit(-1);
    }
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
        thrust::device_vector<int>* input = new thrust::device_vector<int>(data->size());
        thrust::device_vector<double>* retVec = new thrust::device_vector<double>(2160*_numofSites);
        thrust::device_vector<double>* gQuakeAvg = new thrust::device_vector<double>(globalQuakes->size());
        thrust::device_vector<thrust::pair<int, int> >* dConnect = new thrust::device_vector<thrust::pair<int, int> >(_connect->size());
        thrust::copy(data->begin(), data->end(), input->begin());
        thrust::copy(globalQuakes->begin(), globalQuakes->end(), gQuakeAvg->begin());
        thrust::copy(_connect->begin(), _connect->end(), dConnect->begin());

        int gridSize; //the blocksize defined by the configurator
        int blockSize = 512; // the actual grid size needed
        double fitnessAvg=0;
        int fitItr=0;
        gridSize=(_streamSize/_NNParams[7])/(blockSize);
        int host_offset = 0;
        for(int n=0; n<_numOfStreams; n++){

            Net<<<gridSize, blockSize, 0, _stream[n]>>>(device_genetics, convertToKernel(_NNParams),convertToKernel(gQuakeAvg),
                                                        convertToKernel(input),convertToKernel(_siteData),convertToKernel(_answers),
                                                        convertToKernel(dConnect),Kp,_sampleRate,_numofSites, hour,
                                                        meanCh1, meanCh2, meanCh3, stdCh1, stdCh2, stdCh3);

            CUDA_SAFE_CALL(cudaPeekAtLastError());
            std::cerr<<"net completed."<<std::endl;
            int gridSize = (_NNParams[8]/blockSize)+((_NNParams[8]%blockSize) ? 1 : 0);
            thrust::device_vector<double> partial_reduce_sums(gridSize+1);
            reduce_by_block<<<gridSize, blockSize, blockSize*sizeof(double), _stream[n]>>>(device_genetics,
                                                                                           convertToKernel(partial_reduce_sums),
                                                                                           convertToKernel(_NNParams));
            std::cerr<<"reduce by block completed."<<std::endl;
            gridSize=_streamSize/blockSize; //swap in place, every double is a job.
            CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaStreamSynchronize(_stream[n]));
            swapMemory<<<gridSize, blockSize, 0, _stream[n]>>>(device_genetics, host_genetics_device, host_offset);
            std::cerr<<"memory swap completed"<<std::endl;
            CUDA_SAFE_CALL(cudaPeekAtLastError());
                    for(thrust::device_vector<double>::iterator it = partial_reduce_sums.begin();
                    it != partial_reduce_sums.end(); ++it){
                fitnessAvg += *it;
                fitItr++;
            }
            host_offset += _streamSize;
        }
        fitnessAvg = fitnessAvg /fitItr;
        std::cerr<<"the average fitness for this round is: "<<fitnessAvg<<std::endl;

        delete dConnect;
        delete input;
        delete gQuakeAvg;
        delete retVec;
    }
    else{
        std::cerr<<"entered not training version.."<<std::endl;
        typedef std::vector<thrust::pair<int, int> > connectPairMatrix;
        //replace this later
        _best.resize(_NNParams[2]);
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
                CommunityLat += _siteData[j*2]*CommunityMag[j];
                CommunityLon += _siteData[j*2+1]*CommunityMag[j];
            }
            CommunityLat = CommunityLat/_numofSites;
            CommunityLon = CommunityLon/_numofSites;

            for(int j=0; j<_numofSites; j++){ // each site is run independently of others, but shares an output from the previous step
                std::cerr<<"entering site #"<<j<<std::endl;
                double latSite = _siteData[j*2];
                double lonSite = _siteData[j*2+1];
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
                input.resize(_NNParams[3], 0); // number of inputs is 9.
                hidden.resize(_NNParams[4], 0); // for practice sake, lets say each input has its own neuron (might be true!)
                mem.resize(_NNParams[5], 0); // stores the input if gate is high
                memGateOut.resize(_NNParams[5], 0); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
                memGateIn.resize(_NNParams[5], 0);
                memGateForget.resize(_NNParams[5], 0);
                outputs.resize(_NNParams[6], 0); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                    1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
                std::cerr<<"all neuron vectors are sized, all pre-net calculations done."<<std::endl;
                int n =0;
                int startOfInput = 0;
                int startOfHidden = startOfInput +_NNParams[3];
                int startOfMem = startOfHidden + _NNParams[4];
                int startOfMemGateIn = startOfMem + _NNParams[5];
                int startOfMemGateOut = startOfMemGateIn + _NNParams[5];
                int startOfMemGateForget = startOfMemGateOut + _NNParams[5];
                int startOfOutput = startOfMemGateForget + _NNParams[5];
                input[0] = shift((double)(data->at(3600*_sampleRate*j*3 + 0*(3600*_sampleRate)+step)), meanCh1, stdCh1);
                input[1] = normalize((double)(data->at(3600*_sampleRate*j*3 + 1*(3600*_sampleRate)+step)), meanCh2, stdCh2);
                input[2] = normalize((double)(data->at(3600*_sampleRate*j*3 + 2*(3600*_sampleRate)+step)), meanCh3, stdCh3);
                input[3] = shift(GQuakeAvgdist, 40075.1, 0);
                input[4] = shift(GQuakeAvgBearing, 360, 0);
                input[5] = shift(GQuakeAvgMag, 9.5, 0);
                input[6] = shift(Kp, 10, 0);
                input[7] = shift(CommunityDist,40075.1/2, 0);
                input[8] = shift(CommunityBearing, 360, 0);
                //lets reset all neuron values for this new timestep (except memory neurons)
                for(int gate=0; gate<_NNParams[5]; gate++){
                    memGateIn.at(gate) = 0;
                    memGateOut.at(gate) = 0;
                    memGateForget.at(gate) = 0;
                }
                for(int hid=0; hid<_NNParams[4]; hid++){
                    hidden[hid] = 0;
                }
                for(int out=0; out<_NNParams[6]; out++){
                    outputs[out] = 0;
                }
                std::cerr<<"memGate, hidden, and output neurons are zeroed."<<std::endl;
                //now that everything that should be zeroed is zeroed, lets start the network.
                //mem gates & LSTM nodes --
                std::cerr<<"preparing to set the values for memoryGates."<<std::endl;
                for(int gate = 0; gate<_NNParams[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
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
                for (int gate = 0; gate<_NNParams[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
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
                for(int hid=0; hid<_NNParams[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
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

                for(int out =0; out<_NNParams[6]; out++){// add hidden neurons to the output nodes
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
