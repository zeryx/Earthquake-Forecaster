#include <network.h>
#include <kernelDefs.h>
#include <utilFunc.h>
#include <neuroFunc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <assert.h>
//parameter definitions
//    _hostParams.array[0] = number of neurons
//    _hostparams.array[1] = number of weights per individual
//    _hostParams.array[2] = size of individual
//    _hostParams.array[3] = number of input neurons per individual
//    _hostParams.array[4] = number of hidden neurons per individual
//    _hostParams.array[5] = number of memory neurons per individual
//    _hostParams.array[6] = number of memoryGate In nuerons per individual
//    _hostParams.array[7] = number of memoryGate Out neurons per individual
//    _hostParams.array[8] = number of memoryGate Forget neurons per individual
//    _hostParams.array[9] = number of output neurons per individual
//    _hostParams.array[10] = number of individuals in stream
//    _hostParams.array[11] = weights offset
//    _hostParams.array[12] = input offset
//    _hostParams.array[13] = hidden neurons offset
//    _hostParams.array[14] = short term memory neurons offset
//    _hostParams.array[15] = memoryIn neurons offset
//    _hostParams.array[16] = memoryOut neurons offset
//    _hostParams.array[17] = memoryForget neurons offset
//    _hostParams.array[18] = output neurons offset
//    _hostParams.array[19] = fitness offset
//    _hostParams.array[20] = community magnitude offset
//    _hostParams.array[21] = when offset
//    _hostParams.array[22] = howCertain offset
//    _hostParams.array[23] = number of sites
//    _hostParams.array[24] = sample rate
//    _hostParams.array[26] = number of orders
//    _hostParams.array[27] = number of long term memory neurons per individual
//_hostParams.array[28] = long term memory neurons offset


NetworkGenetic::NetworkGenetic(){
    _hostParams.array = new int[30];
    _hostParams.size = 30;
}
NetworkGenetic::~NetworkGenetic(){
    delete[] _hostParams.array;
}

void NetworkGenetic::generateWeights(){
    int blockSize = 512; // number of blocks in the grid
    int gridSize=(_hostParams.array[10]*_hostParams.array[1])/blockSize; //number of weights in stream/blocksize
    size_t global_offset=0;
    size_t device_offset=0;
    std::cerr<<"generating weights.. "<<std::endl;
    std::cerr<<"grid size: "<<gridSize<<std::endl;
    for(int n=0; n<_numOfStreams; n++){//fill the host first.
        if(n%2==0 && n !=0)
            device_offset =0;
        size_t seed;
        FILE *fp;
        fp = std::fopen("/dev/urandom", "r");
        size_t chk;
        chk =std::fread(&seed, 4, 1, fp);
        if(chk <1){std::cerr<<"couldn't read /dev/urandom"<<std::endl; exit(1);}
        std::fclose(fp);
        std::cerr<<"stream number #"<<n+1<<std::endl;
        genWeightsKern<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, _deviceParams, device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[global_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
        global_offset += _streamSize;
        device_offset += _streamSize;
    }
}

void NetworkGenetic::allocateHostAndGPUObjects( size_t deviceRam, size_t hostRam){
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(NetKern, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    long int hostGenSize = hostRam;
    long int deviceGenSize = deviceRam;
    std::cerr<<"total free device ram : "<<deviceRam<<std::endl;
    std::cerr<<"total free host ram : "<<hostRam<<std::endl;

    //this block below makes sure that the number of objects in stream is exactly half of the total amount of allocatable space on the GPU
    _streamSize = deviceGenSize/(sizeof(double)*2);

    while(_streamSize%_hostParams.array[2] || (_streamSize/_hostParams.array[2])&(_streamSize/_hostParams.array[2]-1)) // get the largest number divisible by the individual size, the threads in a block, and the size of a double
        _streamSize= _streamSize -1;

    _streambytes = _streamSize * sizeof(double);
    deviceGenSize = _streambytes * 2;
    assert(deviceGenSize == _streambytes * 2);
    device_genetics.size = _streamSize * 2;

    //this block below makes sure that the allocated host ram for genetics is a number evently divisible by the stream size in bytes.
    host_genetics.size = hostGenSize/sizeof(double);
    _numOfStreams =  host_genetics.size/_streamSize; // number of streams = hostGenSize/streamBytes, device does not store extra weights for simplicity.
    host_genetics.size = _numOfStreams * _streamSize;
    hostGenSize = host_genetics.size * sizeof(double);
    assert(host_genetics.size == _numOfStreams * _streamSize);

    CUDA_SAFE_CALL(cudaHostAlloc((void**)&host_genetics.array, hostGenSize, cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc((void**) &device_genetics.array, deviceGenSize));
    std::fill(host_genetics.array, host_genetics.array+host_genetics.size, 0);
    CUDA_SAFE_CALL(cudaMemset(device_genetics.array, 0, deviceGenSize));
    _stream.resize(_numOfStreams);

    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream.at(i)));
        CUDA_SAFE_CALL( cudaStreamQuery(_stream.at(i)));
    }
    this->confDeviceParams();
}

void NetworkGenetic:: confTestParams(const int &numOfSites, const int &sampleRate){
    this->setParams(23, numOfSites);
    this->setParams(24, sampleRate);
    this->setParams(2, _hostParams.array[0] + _hostParams.array[1] + 1 + 3*15); //size of an individual
}

void NetworkGenetic::confNetParams(const int &numInNeurons, const int &numHiddenNeurons,
                                   const int &numMemoryNeurons, const int &numMemoryIn, const int &numMemoryOut,
                                   const int &numMemoryForget, const int &numOutNeurons){

    this->setParams(0, numInNeurons + numHiddenNeurons + numMemoryNeurons + numMemoryIn + numMemoryOut + numMemoryForget + numOutNeurons);
    this->setParams(3, numInNeurons);
    this->setParams(4, numHiddenNeurons);
    this->setParams(5, numMemoryNeurons);
    this->setParams(6, numMemoryIn);
    this->setParams(7, numMemoryOut);
    this->setParams(8, numMemoryForget);
    this->setParams(9, numOutNeurons);

}

void NetworkGenetic::confOrder(const int &numOrders, const int &numWeights){

    this->setParams(1, numWeights);
    this->setParams(26, numOrders);
    std::cerr<<numOrders<<std::endl;
}


bool NetworkGenetic::loadFromFile(std::ifstream &stream){
    stream.clear();
    stream.seekg(0, stream.beg);
    assert(stream.good());
    std::string item;
    std::cerr<<"preparing to load from file..."<<std::endl;
    size_t entry=0;
    int itr =0;
    while(std::getline(stream, item)){ // each value in the array
        entry++;
    }
    std::cerr<<"number of data points: "<<entry<<std::endl;

    this->allocateHostAndGPUObjects(GetDeviceRamInBytes()*0.85, entry*sizeof(double));
    stream.clear();
    stream.seekg(0, stream.beg);
    assert(stream.good());
    std::cerr.precision(2);
    while(std::getline(stream, item)){ // each value in the array
        host_genetics.array[itr] = std::stod(item);
        if(itr%(host_genetics.size/100) == 0){
            std::cerr<<(float)itr/(float)host_genetics.size<<std::endl;
        }
        itr++;
    }
    std::cerr<<"finished loading from file"<<std::endl;
    return true;
}

void NetworkGenetic::saveToFile(std::ofstream &stream){
    std::cerr<<"saving to file."<<std::endl;
    std::cerr.precision(2);
    for(int itr=0; itr<host_genetics.size; itr++){
        stream<< host_genetics.array[itr]<<"\n";
        if(itr%(host_genetics.size/10) == 0){
            std::cerr<<(float)itr/(float)host_genetics.size<<std::endl;
        }
    }
    CUDA_SAFE_CALL(cudaDeviceReset());
}

void NetworkGenetic::confDeviceParams(){
    this->setParams(10, _streamSize/_hostParams.array[2]); // number of individuals on device
    this->setParams(11, 0);
    this->setParams(12, _hostParams.array[11] + _hostParams.array[10] * _hostParams.array[1]);  // input neurons offset. (weights_offset + numweights*numindividuals)
    this->setParams(13, _hostParams.array[12] + _hostParams.array[10] * _hostParams.array[3]);  // hidden neurons offset. (input_offset +numInputs*numIndividuals)
    this->setParams(14, _hostParams.array[13] + _hostParams.array[10] * _hostParams.array[4]);  // short memory neurons offset. (hidden_offset + numHidden*numIndividuals)
    this->setParams(28, _hostParams.array[14] + _hostParams.array[10] * _hostParams.array[5]);  //long memory neurons offset (smem_offset + numsmem*numIndividuals)
    this->setParams(15, _hostParams.array[14] + _hostParams.array[10] * _hostParams.array[27]);  // memoryIn Gate nodes offset. (lmem_offset + numlMem*numIndividuals)
    this->setParams(16, _hostParams.array[15] + _hostParams.array[10] * _hostParams.array[6]);  // memoryOut Gate nodes offset. (memIn_offset + numMemIn*numIndividuals)
    this->setParams(17, _hostParams.array[16] + _hostParams.array[10] * _hostParams.array[7]);  // memoryForget Gate nodes offset. (memOut_offset + numMemOut*numIndividuals)
    this->setParams(18, _hostParams.array[17] + _hostParams.array[10] * _hostParams.array[8]);  // output neurons offset. (memForget_offset + numMemOut*numIndividuals)
    this->setParams(19, _hostParams.array[18] + _hostParams.array[10] * _hostParams.array[9]);  // fitness offset.
    this->setParams(20, _hostParams.array[20] + _hostParams.array[10] * 1);                     // community magnitude offset
    this->setParams(21, _hostParams.array[20] + _hostParams.array[10] * _hostParams.array[23]); // when offset.
    this->setParams(22, _hostParams.array[21] + _hostParams.array[10] * _hostParams.array[23]); // howCertain offset.

    CUDA_SAFE_CALL(cudaMalloc((void**)&_deviceParams.array, _hostParams.size*sizeof(int)));
    std::cerr<<"number of individuals in stream is: "<<_hostParams.array[10]<<std::endl;
    std::cerr<<"size of connections array: "<<_hostParams.array[26]<<std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(_deviceParams.array, _hostParams.array, _hostParams.size*sizeof(int), cudaMemcpyHostToDevice));
    _deviceParams.size = _hostParams.size;
}

void NetworkGenetic::setParams(int num, int val){
    _hostParams.array[num] = val;
}




void NetworkGenetic::reformatTraining(std::vector<int>&old_input, std::vector<double> &ans, std::vector<double> &sitedata, std::vector<double>&globalquakes, double &kp){ // increase the timestep and reduce resolution, takes too long.
    int trainingSize = 5;
    int * new_input = new int[trainingSize*3*_hostParams.array[23]];
    int *siteOffset = new int[15], *chanOffset = new int[3];
    long long stor[trainingSize*3*_hostParams.array[23]];
    memset(stor, 0, trainingSize*3*_hostParams.array[23]*sizeof(long long));
    siteOffset[0] = 0;
    for(int i=1; i<_hostParams.array[23]; i++){
        siteOffset[i] = trainingSize*3 + siteOffset[i-1];
    }
    chanOffset[0] = 0;
    for(int i=1; i<3; i++){
        chanOffset[i] = trainingSize + chanOffset[i-1];
    }

    for(int step=0; step<trainingSize; step++){
        for(int i=0; i<_hostParams.array[24]*3600/trainingSize; i++){
            for(int k=0; k<_hostParams.array[23]; k++){
                for(int j=0; j<3; j++){
                    stor[siteOffset[k]+chanOffset[j]+step] += old_input.at(k*_hostParams.array[24]*3600*3 + j*_hostParams.array[24]*3600 + step*_hostParams.array[24]*3600/trainingSize+i);
                }
            }
        }
    }
    for(int step=0; step<trainingSize; step++){
        for(int k=0; k<_hostParams.array[23]; k++){
            for(int j=0; j<3; j++){
                stor[siteOffset[k]+chanOffset[j]+step] = stor[siteOffset[k]+chanOffset[j]+step]/(_hostParams.array[24]*3600/trainingSize);
                new_input[siteOffset[k]+chanOffset[j]+step] = stor[siteOffset[k]+chanOffset[j]+step];
            }
        }
    }

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(answers, ans.data(), ans.size()*sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(siteData, sitedata.data(), sitedata.size()*sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(globalQuakes,globalquakes.data(), globalquakes.size()*sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(Kp, &kp, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(inputData, new_input, trainingSize*_hostParams.array[23]*3*sizeof(int), 0,  cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(trainingsize, &trainingSize, sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(site_offset, siteOffset, _hostParams.array[23]*sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(channel_offset, chanOffset, 3*sizeof(int), 0, cudaMemcpyHostToDevice));
    delete[] siteOffset;
    delete[] chanOffset;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void NetworkGenetic::trainForecast(std::vector<double> *ret, int &hour, std::vector<int> &data,
                                   double &Kp, std::vector<double> &globalQuakes,
                                   Order *connections, std::vector<double> &ans, std::vector<double> &siteData){
    //were going to normalize the inputs using v` = v-mean/stdev, so we need mean and stdev for each channel.
    double meanCh[3]{0}, stdCh[3]{0};
    int num=0;
    std::cerr<<"starting training"<<std::endl;
    for(int i=0; i<3600*_hostParams.array[24]; i++){
        for(int j=0; j < _hostParams.array[23]; j++){
            for(int k=0; k<3; k++){
                meanCh[k] += data.at(3600*_hostParams.array[24]*j*3 + k*3600*_hostParams.array[24]+i);
            }

            num++;
        }
    }
    for(int k=0; k<3; k++){
        meanCh[k] = meanCh[k]/num;
        stdCh[k] = sqrt(meanCh[k]);
    }
    //input data from all sites and all channels normalized
    kernelArray<double> retVec, dmeanCh, dstdCh;

    int regBlockSize = 512;
    int regGridSize = (_hostParams.array[10])/regBlockSize;
    retVec.size = 2160*_hostParams.array[23];
    Order *dConnect;

    CUDA_SAFE_CALL(cudaMalloc((void**)&retVec.array, ret->size()*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dmeanCh.array, 3*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dstdCh.array, 3*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dConnect, _hostParams.array[26]*sizeof(Order)));

    CUDA_SAFE_CALL(cudaMemcpy(dConnect, connections, _hostParams.array[26]*sizeof(Order), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dmeanCh.array, meanCh, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dstdCh.array, stdCh, 3*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(retVec.array, 0, retVec.size*sizeof(double)));

    this->reformatTraining(data, ans, siteData,  globalQuakes, Kp);

    size_t host_offset = 0;
    size_t device_offset=0;

    cudaEvent_t waitForLastStream;
    cudaEventCreate(&waitForLastStream);
    std::cerr<<"forcast training loop for hour: "<<hour<<std::endl;
    for(int n=0; n<_numOfStreams; n++){
        if(n%2==0 && n!=0){
            device_offset=0;
            CUDA_SAFE_CALL(cudaEventRecord(waitForLastStream, _stream[n-2]));
        }

        CUDA_SAFE_CALL(cudaStreamWaitEvent(_stream[n], waitForLastStream, 0));
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        std::cerr<<"stream number #"<<n+1<<std::endl;
        CUDA_SAFE_CALL(cudaMemcpyAsync(&device_genetics.array[device_offset], &host_genetics.array[host_offset], _streambytes, cudaMemcpyHostToDevice, _stream[n]));

        NetKern<<<regGridSize, regBlockSize, _hostParams.array[26]*sizeof(Order), _stream[n]>>>(device_genetics,_deviceParams, dConnect, hour, dmeanCh, dstdCh, device_offset);

        CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[host_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
        host_offset += _streamSize;
        device_offset += _streamSize;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cerr.precision(40);
    for(int n=0; n<5; n++){
        std::cerr<<host_genetics.array[_hostParams.array[19]+n]<<std::endl;

        std::cerr<<"first weight is: "<<host_genetics.array[_hostParams.array[11]+n]<<std::endl;

            std::cerr<<"hidden 1 of "<<host_genetics.array[_hostParams.array[13]  + n ]<<std::endl;

            std::cerr<<"output 1 of "<<host_genetics.array[_hostParams.array[18] + n ]<<std::endl;

            for(int i=0; i<_hostParams.array[23]; i++){
                std::cerr<<"for site: "<<i<<std::endl;
                 std::cerr<<"certainty of: "<<host_genetics.array[_hostParams.array[22]+n+i*_hostParams.array[10]]<<std::endl;
                std::cerr<<"when of: "<<host_genetics.array[_hostParams.array[21]+n+i*_hostParams.array[10]]<<std::endl;

            }

        std::cerr<<std::endl;
    }
    CUDA_SAFE_CALL(cudaFree(dConnect));
    CUDA_SAFE_CALL(cudaFree(retVec.array));
    CUDA_SAFE_CALL(cudaFree(dmeanCh.array));
    CUDA_SAFE_CALL(cudaFree(dstdCh.array));

}

void NetworkGenetic::endOfTrial(){
    std::cerr<<"end of trial reached."<<std::endl;

    int regBlockSize = 512;
    int regGridSize = (_hostParams.array[10])/regBlockSize;

    kernelArray<double> partial_reduce_sums;
    double *hfitnessAvg, *dfitnessAvg;
    int *hparentChildCutoff, *dparentChildCutoff;
    int *evoGridSize;
    partial_reduce_sums.size = regBlockSize*_numOfStreams;
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&hparentChildCutoff, _numOfStreams*sizeof(int), cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&hfitnessAvg, _numOfStreams*sizeof(double), cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc((void**)&partial_reduce_sums.array, regBlockSize*_numOfStreams*sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&evoGridSize, _numOfStreams*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dparentChildCutoff, _numOfStreams*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dfitnessAvg, _numOfStreams*sizeof(double)));
    partial_reduce_sums.size = (regGridSize);

    size_t host_offset = 0;
    size_t device_offset=0;
    size_t *seed = new size_t[_numOfStreams];

    for(int i=0; i<_numOfStreams; i++){//set random numbers for evolution seed
        FILE *fp;
        fp = std::fopen("/dev/urandom", "r");
        size_t chk;
        chk =std::fread(&seed[i], 4, 1, fp);
        if(chk <1){std::cerr<<"couldn't read /dev/urandom"<<std::endl; exit(1);}
        std::fclose(fp);
    }

    cudaEvent_t waitForLastStream;
    cudaEventCreate(&waitForLastStream);

    for(int n=0; n<_numOfStreams; n++){
        if(n%2==0 && n!=0){
            device_offset=0;
            CUDA_SAFE_CALL(cudaEventRecord(waitForLastStream, _stream[n-2]));
        }

        CUDA_SAFE_CALL(cudaStreamWaitEvent(_stream[n], waitForLastStream, 0));
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        std::cerr<<"stream number #"<<n+1<<std::endl;

        CUDA_SAFE_CALL(cudaMemcpyAsync(&device_genetics.array[device_offset], &host_genetics.array[host_offset], _streambytes, cudaMemcpyHostToDevice, _stream[n]));
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        for(int k=2; k<= _hostParams.array[10]; k<<= 1){
            for(int j =k>>1; j>0; j=j>>1){
                bitonicSortKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, j, k, device_offset);
            }
        }
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        reduceFirstKern<<<regGridSize, regBlockSize, regBlockSize*sizeof(double), _stream[n]>>>(device_genetics, partial_reduce_sums, _deviceParams, device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        reduceSecondKern<<<1, 1, 0, _stream[n]>>>(partial_reduce_sums, _deviceParams, &dfitnessAvg[n]);
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        normalizeKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, &dfitnessAvg[n], device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        CUDA_SAFE_CALL(cudaMemcpyAsync(&hfitnessAvg[n], &dfitnessAvg[n], sizeof(double), cudaMemcpyDeviceToHost, _stream[n]));
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        cutoffKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams,  &dparentChildCutoff[n], &evoGridSize[n], &dfitnessAvg[n], device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        CUDA_SAFE_CALL(cudaMemcpyAsync(&hparentChildCutoff[n], &dparentChildCutoff[n], sizeof(int), cudaMemcpyDeviceToHost, _stream[n]));
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        evolutionKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, &dparentChildCutoff[n], &evoGridSize[n], seed[n], device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());

        CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[host_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        host_offset += _streamSize;
        device_offset += _streamSize;
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cerr<<"average fitness for first stream is: "<< hfitnessAvg[0]<<std::endl;

    CUDA_SAFE_CALL(cudaFree(partial_reduce_sums.array));
    CUDA_SAFE_CALL(cudaFreeHost(hfitnessAvg));
    CUDA_SAFE_CALL(cudaFreeHost(hparentChildCutoff));
    CUDA_SAFE_CALL(cudaFree(dfitnessAvg));
    CUDA_SAFE_CALL(cudaFree(evoGridSize));
    CUDA_SAFE_CALL(cudaFree(dparentChildCutoff));
    CUDA_SAFE_CALL(cudaFree(device_genetics.array));
    delete[] seed;
}

void NetworkGenetic::challengeForecast(std::vector<double> *ret, int &hour, std::vector<int> &data, double &Kp,
                                       std::vector<double> &globalQuakes, Order *connections, std::vector<double> &siteData){
    //were going to normalize the inputs using v` = v-mean/stdev, so we need mean and stdev for each channel.
    //    double meanCh[3]{0}, stdCh[3]{0};
    //    int num=0;
    //    for(int i=0; i<3600*_hostParams.array[24]; i++){
    //        for(int j=0; j < _hostParams.array[23]; j++){
    //            for(int k=0; k<3; k++){
    //                meanCh[k] += data.at(3600*_hostParams.array[24]*j*3 + k*3600*_hostParams.array[24]+i);
    //            }
    //            num++;
    //        }
    //    }
    //    for(int k=0; k<3; k++){
    //        meanCh[k] = meanCh[k]/num;
    //        stdCh[k] = sqrt(meanCh[k]);
    //    }
    //    std::cerr<<"entered not training version.."<<std::endl;
    //    //replace this later
    //    //        _best.resize(_hostParams.array[1]);
    //    //        for(std::vector<double>::iterator it = _best.begin(); it != _best.end(); ++it){
    //    //            std::srand(std::time(NULL)+*it);
    //    //            *it = (double)(std::rand())/(RAND_MAX);
    //    //        }
    //    std::cerr<<"example best vector has been set."<<std::endl;
    //    double CommunityLat = 0;
    //    double CommunityLon = 0;
    //    std::vector<double> When(_hostParams.array[23], 0);
    //    std::vector<double> HowCertain(_hostParams.array[23],0);
    //    std::vector<double> CommunityMag(_hostParams.array[23], 1); //give all sites equal mag to start, this value is [0,1]
    //    std::cerr<<"all output vectors created and initialized."<<std::endl;
    //    for(int step=0; step<3600*_hostParams.array[24]; step++){
    //        for(int j=0; j<_hostParams.array[23]; j++){ //sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
    //            CommunityLat += siteData.at(j*2)*CommunityMag[j];
    //            CommunityLon += siteData.at(j*2+1)*CommunityMag[j];
    //        }
    //        CommunityLat = CommunityLat/_hostParams.array[23];
    //        CommunityLon = CommunityLon/_hostParams.array[23];

    //        for(int j=0; j<_hostParams.array[23]; j++){ // each site is run independently of others, but shares an output from the previous step
    //            double latSite = siteData.at(j*2);
    //            double lonSite = siteData.at(j*2+1);
    //            double avgLatGQuake = globalQuakes.at(0);
    //            double avgLonGQuake = globalQuakes.at(1);
    //            double GQuakeAvgMag = globalQuakes.at(3);
    //            double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
    //            double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
    //            double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
    //            double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
    //            std::vector<double> input;
    //            std::vector<double> hidden, output, mem, memGateOut, memGateIn, memGateForget;
    //            //replace these with real connections, num of inputs, and num of hidden & memory neurons (mem neurons probably accurate)
    //            input.resize(_hostParams.array[2], 0); // number of inputs is 9.
    //            hidden.resize(_hostParams.array[10], 0); // for practice sake, lets say each input has its own neuron (might be true!)
    //            mem.resize(_hostParams.array[11], 0); // stores the input if gate is high
    //            memGateOut.resize(_hostParams.array[11], 0); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
    //            memGateIn.resize(_hostParams.array[11], 0);
    //            memGateForget.resize(_hostParams.array[11], 0);
    //            output.resize(_hostParams.array[12], 0); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
    //                    1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
    //            int n =0;
    //            for(int k=0; k<3; k++){
    //                input[k] = normalize((double)(data.at(3600*_hostParams.array[24]*j*3 + k*(3600*_hostParams.array[24])+step)), meanCh[k], stdCh[k]);

    //            }
    //            input[3] = shift(GQuakeAvgdist, 40075.1, 0);
    //            input[4] = shift(GQuakeAvgBearing, 360, 0);
    //            input[5] = shift(GQuakeAvgMag, 9.5, 0);
    //            input[6] = shift(Kp, 10, 0);
    //            input[7] = shift(CommunityDist,40075.1/2, 0);
    //            input[8] = shift(CommunityBearing, 360, 0);
    //            //lets reset all neuron values for this new timestep (except memory neurons)
    //            for(int itr=0; itr< _hostParams.array[26]; itr++){//every order is sequential and run after the previous order to massively simplify the workload in this kernel.

    //                //set stuff to zero
    //                if(connections[itr].first.def == typeInput && connections[itr].second.def == typeZero)
    //                    neuroZero(input[connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeZero)
    //                    neuroZero(hidden[connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeMemGateIn && connections[itr].second.def == typeZero)
    //                    neuroZero(memGateIn[+connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeMemGateOut && connections[itr].second.def == typeZero)
    //                    neuroZero(memGateOut[connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeMemGateForget && connections[itr].second.def == typeZero)
    //                    neuroZero(memGateForget[connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeMemory && connections[itr].second.def == typeZero)
    //                    neuroZero(mem[connections[itr].first.id]);

    //                else if(connections[itr].first.def == typeOutput && connections[itr].second.def == typeZero)
    //                    neuroZero(output[connections[itr].first.id]);

    //                //first->second summations
    //                else if(connections[itr].first.def == typeInput && connections[itr].second.def == typeHidden)
    //                    neuroSum(hidden[connections[itr].second.id],
    //                            (input[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeInput && connections[itr].second.def == typeMemGateIn)
    //                    neuroSum(memGateIn[ + connections[itr].second.id],
    //                            (input[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeInput && connections[itr].second.def == typeMemGateOut)
    //                    neuroSum(memGateIn[connections[itr].second.id],
    //                            (input[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeInput && connections[itr].second.def == typeMemGateForget)
    //                    neuroSum(memGateForget[connections[itr].second.id],
    //                            (input[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeHidden)
    //                    neuroSum(hidden[connections[itr].second.id],
    //                            (hidden[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeMemGateIn)
    //                    neuroSum(memGateIn[connections[itr].second.id],
    //                            (hidden[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeMemGateOut)
    //                    neuroSum(memGateIn[connections[itr].second.id],
    //                            (hidden[connections[itr].first.id])*(_best[n++]));

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeMemGateForget)
    //                    neuroSum(memGateForget[connections[itr].second.id],
    //                            (hidden[connections[itr].first.id])*(_best[n++]));

    //                //memory gates
    //                else if(connections[itr].first.def == typeInput && connections[itr].second.def == typeMemory && connections[itr].third.def == typeMemGateIn)
    //                    neuroMemGate(memGateIn[connections[itr].third.id],
    //                            input[connections[itr].first.id],
    //                            mem[connections[itr].second.id], 0.5);

    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeMemory && connections[itr].third.def == typeMemGateIn)
    //                    neuroMemGate(memGateIn[+connections[itr].third.id],
    //                            hidden[connections[itr].first.id],
    //                            mem[connections[itr].second.id], 0.5);

    //                else if(connections[itr].first.def == typeOutput && connections[itr].second.def == typeMemory && connections[itr].third.def == typeMemGateIn)
    //                    neuroMemGate(memGateIn[connections[itr].third.id],
    //                            output[connections[itr].first.id],
    //                            mem[connections[itr].second.id], 0.5);

    //                else if(connections[itr].first.def == typeMemory && connections[itr].second.def == typeHidden && connections[itr].third.def == typeMemGateOut)
    //                    neuroMemGate(memGateOut[connections[itr].third.id],
    //                            mem[connections[itr].first.id],
    //                            hidden[connections[itr].second.id], 0.5);

    //                else if(connections[itr].first.def == typeMemory && connections[itr].second.def == typeOutput && connections[itr].third.def == typeMemGateOut)
    //                    neuroMemGate(memGateOut[connections[itr].third.id],
    //                            mem[connections[itr].first.id],
    //                            output[connections[itr].second.id], 0.5);

    //                else if(connections[itr].first.def == typeMemory && connections[itr].second.def == typeMemGateForget)
    //                    neuroMemForget(memGateForget[connections[itr].second.id],
    //                            mem[connections[itr].first.id], 0.5);
    //                //bias
    //                else if(connections[itr].first.def == typeBias && connections[itr].second.def == typeHidden)
    //                    neuroSum(hidden[connections[itr].second.id], (1*(_best[n++])));

    //                else if(connections[itr].first.def == typeBias && connections[itr].second.def == typeMemGateIn)
    //                    neuroSum(memGateIn[connections[itr].second.id], (1*(_best[n++])));

    //                else if(connections[itr].first.def == typeBias && connections[itr].second.def == typeMemGateOut)
    //                    neuroSum(memGateIn[connections[itr].second.id], (1*(_best[n++])));

    //                else if(connections[itr].first.def == typeBias && connections[itr].second.def == typeMemGateForget)
    //                    neuroSum(memGateForget[connections[itr].second.id], (1*(_best[n++])));

    //                else if(connections[itr].first.def == typeBias && connections[itr].second.def == typeOutput)
    //                    neuroSum(output[connections[itr].second.id], (1*(_best[n++])));

    //                //squashing
    //                else if(connections[itr].first.def == typeHidden && connections[itr].second.def == typeSquash)
    //                    neuroSquash(hidden[connections[itr].second.id]);

    //                else if(connections[itr].first.def == typeMemGateIn && connections[itr].second.def == typeSquash)
    //                    neuroSquash(memGateIn[ + connections[itr].second.id]);

    //                else if(connections[itr].first.def == typeMemGateOut && connections[itr].second.def == typeSquash)
    //                    neuroSquash(memGateIn[connections[itr].second.id]);

    //                else if(connections[itr].first.def == typeMemGateForget && connections[itr].second.def == typeSquash)
    //                    neuroSquash(memGateForget[connections[itr].second.id]);

    //                else if(connections[itr].first.def == typeOutput && connections[itr].second.def == typeSquash)
    //                    neuroSquash(output[connections[itr].second.id]);

    //            }


    //            When[j] += output[0]*((2160-hour)-hour)+2160-hour; //return when back to an integer value (adjust to fit within boundaries)
    //            HowCertain[j] += output[1];
    //            CommunityMag[j] =  output[2]; // set the next sets communityMag = output #3.
    //        }
    //    }
    //        float maxCertainty=0;
    //        float whenGuess=0;
    //        float guessLat=0;
    //        float guessLon=0;
    //        for(int j=0; j<_hostParams.array[23]; j++){
    //            if(HowCertain[j] > maxCertainty){
    //                maxCertainty = HowCertain[j];
    //                whenGuess = When[j];
    //                guessLat = siteData.at(j*2);
    //                guessLon = siteData.at(j*2+1);
    //            }
    //        }

    //        int whenAns = (int)_answers[1]-hour;
    //        double oldFit = ret->at(0);
    //        ret->at(0) = scoreFunc(whenGuess, whenAns, guessLat, guessLon, ansLat, ansLon, oldFit);//larger is better, negative numbers are impossible.
}
