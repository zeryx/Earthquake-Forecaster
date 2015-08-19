﻿#include <network.h>
#include <kernelDefs.h>
#include <getsys.h>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <assert.h>


NetworkGenetic::NetworkGenetic(const int &numInputNodes, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numWeights, std::vector< std::pair<int, int> >&connections){

    _hostParams.array = new int[23];
    _hostParams.size=23;
    _hostParams.array[0] = numInputNodes + numHiddenNeurons + numMemoryNeurons*4 + numOutNeurons;
    _hostParams.array[1] = numWeights;
  _hostParams.array[2] = _hostParams.array[0] + _hostParams.array[1] + 2; //1 for community mag, 1 for fitness
    _hostParams.array[3] = numInputNodes;
    _hostParams.array[4] = numHiddenNeurons;
    _hostParams.array[5] =numMemoryNeurons;            //memory neurons per individual
    _hostParams.array[6] =numMemoryNeurons;            //memoryIn neurons per individual
    _hostParams.array[7] =numMemoryNeurons;            //memoryOut neurons per individual
    _hostParams.array[8] = numMemoryNeurons;       //memoryForget neurons per individual
    _hostParams.array[9] =numOutNeurons;           //output neurons per individual
    //    _hostParams.array[10] = number of individuals in stream
    //    _hostParams.array[11] = weights offset
    //    _hostParams.array[12] = input offset
    //    _hostParams.array[13] = hidden neurons offset
    //    _hostParams.array[14] = memory neurons offset
    //    _hostParams.array[15] = memoryIn neurons offset
    //    _hostParams.array[16] = memoryOut neurons offset
    //    _hostParams.array[17] = memoryForget neurons offset
    //    _hostParams.array[18] = output neurons offset
    //    _hostParams.array[19] = community magnitude offset
    //    _hostParams.array[19] = fitness offset

    _connect = &connections;
}

void NetworkGenetic::generateWeights(){
    int blockSize = 512; // number of blocks in the grid
    int gridSize=(_streamSize - _hostParams.array[10]*_hostParams.array[0])/blockSize; //number of weights in stream/blocksize
    size_t global_offset=0;
    size_t device_offset=0;
    std::cerr<<"blocks in this grid: "<<gridSize<<std::endl;
    for(int n=0; n<_numOfStreams; n++){//fill the host first.
        if(n%4==0 && n !=0)
            device_offset =0;
        uint32_t seed;
        FILE *fp;
        fp = std::fopen("/dev/urandom", "r");
        size_t chk;
        chk =std::fread(&seed, 4, 1, fp);
        if(chk <1){std::cerr<<"couldn't read /dev/urandom"<<std::endl; exit(1);}
        std::fclose(fp);
        std::cerr<<"stream number #"<<n<<std::endl;
        std::cerr<<"seed is:"<<seed<<std::endl;
        std::cerr<<"global offset: "<<global_offset<<std::endl;
        std::cerr<<"device offset: "<<device_offset<<std::endl<<std::endl;
        genWeightsKern<<< gridSize, blockSize, 0, _stream[n]>>>(device_genetics, seed, _deviceParams, device_offset);
        CUDA_SAFE_CALL(cudaPeekAtLastError());
        CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[global_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
        global_offset += _streamSize;
        device_offset += _streamSize;
    }
    //    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //    for(int i=(_hostParams.array[1]*_hostParams.array[10]-10000); i<_hostParams.array[1]*_hostParams.array[10]; i++){
    //        std::cerr<<host_genetics.array[i]<<std::endl;
    //    }
    //    cudaDeviceReset();
    //    exit(1);
}


void NetworkGenetic::allocateHostAndGPUObjects( float pMax, size_t deviceRam, size_t hostRam){
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    size_t totalHost = hostRam*pMax;
    size_t totalDevice = deviceRam*pMax;
    std::cerr<<"total free device ram : "<<deviceRam<<std::endl;
    std::cerr<<"total free host ram : "<<hostRam<<std::endl;
    //each memory block is divisible by the size of an individual, the size of double, and the blocksize
    while(totalHost%_hostParams.array[2] || totalHost%sizeof(double) || totalHost%(sizeof(double)*4) || totalHost%512) // get the largest number divisible by the individual size, the threads in a block, and the size of a double
        totalHost= totalHost -1;
    while(totalDevice%_hostParams.array[2] || totalDevice%sizeof(double) || totalDevice%(sizeof(double)*4) || totalDevice%512)
        totalDevice = totalDevice -1;

    _streamSize = totalDevice/(sizeof(double)*4);
    _streambytes = _streamSize*sizeof(double);

    assert(_streambytes == _streamSize*sizeof(double));
    _numOfStreams = ceil((totalHost/sizeof(double))/_streamSize); // number of streams = totalHost/streamBytes, device does not store extra weights for simplicity.
    assert(_streambytes * _numOfStreams <= totalDevice+totalHost);
    std::cerr<<"number of streams: "<<_numOfStreams<<std::endl;
    device_genetics.size = (totalDevice)/sizeof(double);
    host_genetics.size = totalHost/sizeof(double);
    std::cerr<<"device ram to allocate: "<<totalDevice<<std::endl;
    std::cerr<<"host ram to allocate: "<<totalHost<<std::endl;
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&host_genetics.array, totalHost, cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc((void**) &device_genetics.array, totalDevice));
    std::cerr<<"all allocated, moving on."<<std::endl;
    _stream.resize(_numOfStreams);
    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream.at(i)));
        CUDA_SAFE_CALL( cudaStreamQuery(_stream.at(i)));
    }
    this->setParams();
}
bool NetworkGenetic::init(int sampleRate, int SiteNum, std::vector<double> *siteData){
    _sampleRate = sampleRate;
    _numofSites = SiteNum;
    _siteData = siteData;
    _istraining = false;
    return true;
}

void NetworkGenetic::setParams(){
    _hostParams.array[10] = _streamSize/_hostParams.array[2];
    _hostParams.array[11] = 0;
    _hostParams.array[12] = _hostParams.array[11] + _hostParams.array[10] * _hostParams.array[1];  // input neurons offset. (weights_offset + numweights*numindividuals)
    _hostParams.array[13] = _hostParams.array[12] + _hostParams.array[10] * _hostParams.array[3]; // hidden neurons offset. (input_offset +numInputs*numIndividuals)
    _hostParams.array[14] = _hostParams.array[13] + _hostParams.array[10] * _hostParams.array[4]; // memory neurons offset. (hidden_offset + numHidden*numIndividuals)
    _hostParams.array[15] = _hostParams.array[14] + _hostParams.array[10] * _hostParams.array[5]; // memoryIn Gate nodes offset. (mem_offset + numMem*numIndividuals)
    _hostParams.array[16] = _hostParams.array[15] + _hostParams.array[10] * _hostParams.array[6];// memoryOut Gate nodes offset. (memIn_offset + numMemIn*numIndividuals)
    _hostParams.array[17] = _hostParams.array[16] + _hostParams.array[10] * _hostParams.array[7];
    _hostParams.array[18] = _hostParams.array[17] + _hostParams.array[10] *_hostParams.array[8]; // output neurons offset. (memOut_offset + numMemOut*numIndividuals)
    _hostParams.array[19] = _hostParams.array[18] + _hostParams.array[10] *_hostParams.array[9]; // fitness offset.
    std::cerr<<"allocating params memory"<<std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_deviceParams.array, _hostParams.size*sizeof(int)));
    std::cerr<<"setting params memory"<<std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(_deviceParams.array, _hostParams.array, _hostParams.size*sizeof(int), cudaMemcpyHostToDevice));
    _deviceParams.size = _hostParams.size;
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

void NetworkGenetic::reformatTraining(kernelArray<int> old_input, kernelArray<int> new_input){ // increase the timestep and reduce resolution, takes too long.
    int blocksize = 512;
    size_t interleaveGridSize = (3600)/blocksize;
    int *siteOffset = new int[_numofSites], *chanOffset = new int[3];

    siteOffset[0] = 0;
    for(int i=1; i<_numofSites; i++){
        siteOffset[i] = 3600*3 + siteOffset[i-1];
    }
    chanOffset[0] = 0;
    for(int i=1; i<3; i++){
        chanOffset[i] = 3600 + chanOffset[i-1];
    }
    CUDA_SAFE_CALL(cudaMalloc((void**)&_site_offset,_numofSites*sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_channel_offset, 3*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(_site_offset, siteOffset, _numofSites*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_channel_offset, chanOffset, 3*sizeof(int), cudaMemcpyHostToDevice));
    interKern<<<interleaveGridSize, blocksize>>>(old_input, new_input ,_site_offset, _channel_offset, _sampleRate, _numofSites);
    CUDA_SAFE_CALL(cudaPeekAtLastError());
}

void NetworkGenetic::forecast(std::vector<double> *ret, int &hour, std::vector<int> *data, double &Kp, std::vector<double> *globalQuakes)
{
    //were going to normalize the inputs using v` = v-mean/stdev, so we need mean and stdev for each channel.
    double meanCh[3]{0}, stdCh[3]{0};
    int num=0;
    std::cerr<<"right before mean & std calc"<<std::endl;
    for(int i=0; i<3600*_sampleRate; i++){
        for(int j=0; j < _numofSites; j++){
            for(int k=0; k<3; k++){
            meanCh[k] += data->at(3600*_sampleRate*j*3 + k*3600*_sampleRate+i);
            }
            num++;
        }
    }
    for(int k=0; k<3; k++){
    meanCh[k] = meanCh[k]/num;
    stdCh[k] = sqrt(meanCh[k]);
    }
    std::cerr<<"channels std and mean calculated"<<std::endl;
    //input data from all sites and all channels normalized
    if(_istraining == true){
        std::cerr<<"beginning of training"<<std::endl;
        kernelArray<double>retVec, gQuakeAvg, answers, siteData, partial_reduce_sums, dmeanCh, dstdCh;
        kernelArray<int> rawInput, correctedInput;
        kernelArray<std::pair<int, int> > dConnect;
        int regBlockSize = 512;
        size_t reduceGridSize = (_streamSize/_hostParams.array[2])/regBlockSize + (((_streamSize/_hostParams.array[2])%regBlockSize) ? 1 : 0);
        size_t netGridSize = (_streamSize/_hostParams.array[2])/regBlockSize;
        correctedInput.size = data->size();
        retVec.size = 2160*_numofSites;
        gQuakeAvg.size = globalQuakes->size();
        answers.size = _answers.size();
        dConnect.size = _connect->size();
        siteData.size = _siteData->size();
        partial_reduce_sums.size = _numOfStreams*(regBlockSize+1);
        CUDA_SAFE_CALL(cudaMalloc((void**)&rawInput.array, data->size()*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&correctedInput.array, 3600*3*_numofSites*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&retVec.array, ret->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&gQuakeAvg.array, globalQuakes->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&answers.array, _answers.size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dConnect.array, _connect->size()*sizeof(std::pair<int, int>)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&siteData.array, _siteData->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&partial_reduce_sums.array, _numOfStreams*(regBlockSize+1)*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dmeanCh.array, 3*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dstdCh.array, 3*sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpy(rawInput.array, data->data(), data->size()*sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(gQuakeAvg.array, globalQuakes->data(), globalQuakes->size()*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dConnect.array, _connect->data(), _connect->size()*sizeof(std::pair<int, int>), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(siteData.array, _siteData->data(), _siteData->size()*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(answers.array, _answers.data(), _answers.size()*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dmeanCh.array, meanCh, 3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dstdCh.array, stdCh, 3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(correctedInput.array, 0, 3600*3*_numofSites*sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(retVec.array, 0, retVec.size*sizeof(double)));
        CUDA_SAFE_CALL(cudaMemset(partial_reduce_sums.array, 0, partial_reduce_sums.size*sizeof(double)));
        std::cerr<<"synchronizing device before correcting input data.."<<std::endl;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::cerr<<"device synchronized, correcting input data."<<std::endl;
        this->reformatTraining(rawInput, correctedInput);
        CUDA_SAFE_CALL(cudaFree(rawInput.array));
        std::cerr<<"input data corrected, running main sequence."<<std::endl;
        double fitnessAvg=0;
        int fitItr=0;
        size_t host_offset = 0;
        size_t device_offset=0;
        for(int n=0; n<_numOfStreams; n++){
            if(n%4==0 && n!=0){
                device_offset=0;
            }
            if(n>=4)
                CUDA_SAFE_CALL(cudaStreamSynchronize(_stream[n-4]));

            std::cerr<<"stream number: "<<n<<std::endl;
            std::cerr<<"host offset: "<<host_offset<<std::endl;
            std::cerr<<"device offset: "<<device_offset<<std::endl;
            CUDA_SAFE_CALL(cudaMemcpyAsync(&device_genetics.array[device_offset], &host_genetics.array[host_offset], _streambytes, cudaMemcpyHostToDevice, _stream[n]));
            CUDA_SAFE_CALL(cudaPeekAtLastError());
                NetKern<<<netGridSize, regBlockSize, regBlockSize*sizeof(float)*3*_numofSites, _stream[n]>>>(device_genetics, _deviceParams, gQuakeAvg, correctedInput, siteData, answers, dConnect, Kp,_sampleRate,_numofSites, _site_offset, _channel_offset, hour, dmeanCh, dstdCh, device_offset);
                CUDA_SAFE_CALL(cudaPeekAtLastError());
            //            reduceKern<<<reduceGridSize, regBlockSize, regBlockSize*sizeof(double), _stream[n]>>>(device_genetics, partial_reduce_sums, _deviceParams, device_offset, reduceGridSize*n);
            //            CUDA_SAFE_CALL(cudaPeekAtLastError());
            CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[host_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
            CUDA_SAFE_CALL(cudaPeekAtLastError());
            //            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            //                        for(int itr =0; itr<partial_reduce_sums.size; itr++){
            //                            fitnessAvg += partial_reduce_sums.array[itr];
            //                            fitItr++;
            //                        }
            host_offset += _streamSize;
            device_offset += _streamSize;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        for(int i=_hostParams.array[12]; i<_hostParams.array[12]+_hostParams.array[3]*_hostParams.array[10]; i++){
            std::cerr<<i<<" is: "<<host_genetics.array[i]<<std::endl;
        }
        cudaDeviceReset();
        exit(1);
        //        fitnessAvg = fitnessAvg /fitItr;
        //        std::cerr<<"the average fitness for this round is: "<<fitnessAvg<<std::endl;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        CUDA_SAFE_CALL(cudaFree(dConnect.array));
        CUDA_SAFE_CALL(cudaFree(correctedInput.array));
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
        _best.resize(_hostParams.array[1]);
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
            for(int j=0; j<_numofSites; j++){ //sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
                CommunityLat += _siteData->at(j*2)*CommunityMag[j];
                CommunityLon += _siteData->at(j*2+1)*CommunityMag[j];
            }
            CommunityLat = CommunityLat/_numofSites;
            CommunityLon = CommunityLon/_numofSites;

            for(int j=0; j<_numofSites; j++){ // each site is run independently of others, but shares an output from the previous step
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
                input.resize(_hostParams.array[2], 0); // number of inputs is 9.
                hidden.resize(_hostParams.array[10], 0); // for practice sake, lets say each input has its own neuron (might be true!)
                mem.resize(_hostParams.array[11], 0); // stores the input if gate is high
                memGateOut.resize(_hostParams.array[11], 0); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
                memGateIn.resize(_hostParams.array[11], 0);
                memGateForget.resize(_hostParams.array[11], 0);
                outputs.resize(_hostParams.array[12], 0); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                    1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
                int n =0;
                int startOfInput = 0;
                int startOfHidden = startOfInput +_hostParams.array[2];
                int startOfMem = startOfHidden + _hostParams.array[10];
                int startOfMemGateIn = startOfMem + _hostParams.array[11];
                int startOfMemGateOut = startOfMemGateIn + _hostParams.array[11];
                int startOfMemGateForget = startOfMemGateOut + _hostParams.array[11];
                int startOfOutput = startOfMemGateForget + _hostParams.array[11];
                for(int k=0; k<3; k++){
                    input[k] = normalize((double)(data->at(3600*_sampleRate*j*3 + k*(3600*_sampleRate)+step)), meanCh[k], stdCh[k]);

                }
                input[3] = shift(GQuakeAvgdist, 40075.1, 0);
                input[4] = shift(GQuakeAvgBearing, 360, 0);
                input[5] = shift(GQuakeAvgMag, 9.5, 0);
                input[6] = shift(Kp, 10, 0);
                input[7] = shift(CommunityDist,40075.1/2, 0);
                input[8] = shift(CommunityBearing, 360, 0);
                //lets reset all neuron values for this new timestep (except memory neurons)
                for(int gate=0; gate<_hostParams.array[11]; gate++){
                    memGateIn.at(gate) = 0;
                    memGateOut.at(gate) = 0;
                    memGateForget.at(gate) = 0;
                }
                for(int hid=0; hid<_hostParams.array[10]; hid++){
                    hidden[hid] = 0;
                }
                for(int out=0; out<_hostParams.array[12]; out++){
                    outputs[out] = 0;
                }
                //now that everything that should be zeroed is zeroed, lets start the network.
                //mem gates & LSTM nodes --
                for(int gate = 0; gate<_hostParams.array[11]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateIn
                        if(it->second == gate+startOfMemGateIn && it->first < startOfHidden){ //for inputs
                            memGateIn.at(gate) += input[it->first-startOfInput]*_best[n++]; // memGateIn vect starts at 0
                        }
                        else if(it->second == gate+startOfMemGateIn && it->first >startOfHidden && it->first < startOfMem){//for hidden neurons
                            memGateIn.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateOut
                        if(it->second == gate+startOfMemGateOut && it->first < startOfHidden){//for inputs
                            memGateOut.at(gate) += input[it->first-startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateOut && it->first >startOfHidden && it->first <startOfMem){//for hidden neurons
                            memGateOut.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for  memGateForget
                        if(it->second == gate+startOfMemGateForget && it->first < startOfHidden){//for inputs
                            memGateForget.at(gate) += input[it->first - startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateForget && it->first >startOfHidden && it->first <startOfMem){//for hidden neurons
                            memGateForget.at(gate) += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    memGateIn.at(gate) = ActFunc(memGateIn.at(gate));
                    memGateOut.at(gate) = ActFunc(memGateOut.at(gate));
                    memGateForget.at(gate) = ActFunc(memGateForget.at(gate));

                }
                //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
                for (int gate = 0; gate<_hostParams.array[11]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
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
                }

                // hidden neuron nodes --
                for(int hid=0; hid<_hostParams.array[10]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
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
                }
                //output nodes --

                for(int out =0; out<_hostParams.array[12]; out++){// add hidden neurons to the output nodes
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                        if(it->second == out+startOfOutput){
                            outputs[out] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    outputs[out] += 1*_best[n++]; // add bias
                    outputs[out] = ActFunc(outputs[out]);// then squash it.
                }


                When[j] += outputs[0]*((2160-hour)-hour)+2160-hour; //return when back to an integer value (adjust to fit within boundaries)
                HowCertain[j] += outputs[1];
                CommunityMag[j] =  outputs[2]; // set the next sets communityMag = output #3.
            }
        }
        for(int j=0; j<_numofSites; j++){ // each site has its own when and howcertain vector
            When[j] = When[j]/3600*_sampleRate;
            HowCertain[j] = HowCertain[j]/3600*_sampleRate;
            std::cerr<<"When for site:"<<j<<" is: "<<When[j]<<std::endl;
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
