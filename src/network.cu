#include <network.h>
#include <kernelDefs.h>
#include <getsys.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <assert.h>


NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons, const int &numMemoryIn,
                               const int &numMemoryOut, const int &numMemoryForget,
                               const int &numOutNeurons, const int &numWeights,  std::vector< std::pair<hcon, hcon> >&connections){

    _hostParams.array = new int[27];
    _hostParams.size=27;
    _hostParams.array[0] = numInNeurons + numHiddenNeurons + numMemoryNeurons + numMemoryIn + numMemoryOut + numMemoryForget + numOutNeurons;
    _hostParams.array[1] = numWeights;
    //    _hostParams.array[2] = _hostParams.array[0] + _hostParams.array[1] + 2 + 3*_hostParams.array[23]; //1*numOfSites for community mag, 1*numOfSites for When, 1*numOfSites for HowCertain,  1 for fitness, and 1 for age.
    _hostParams.array[3] = numInNeurons;
    _hostParams.array[4] = numHiddenNeurons;
    _hostParams.array[5] =numMemoryNeurons;            //memory neurons per individual
    _hostParams.array[6] =numMemoryIn;            //memoryIn neurons per individual
    _hostParams.array[7] =numMemoryOut;            //memoryOut neurons per individual
    _hostParams.array[8] = numMemoryForget;       //memoryForget neurons per individual
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
    //    _hostParams.array[19] = fitness offset
    //    _hostParams.array[20] = community magnitude offset
    //    _hostParams.array[21] = when offset
    //    _hostParams.array[22] = howCertain offset
    //    _hostParams.array[23] = number of sites
    //    _hostParams.array[24] = sample rate
    //    _hostParams.array[25] = age offset
    _connect = &connections;
    _hostParams.array[26] = _connect->size(); //number of connections
}

void NetworkGenetic::generateWeights(){
    int blockSize = 512; // number of blocks in the grid
    int gridSize=(_hostParams.array[10]*_hostParams.array[1])/blockSize; //number of weights in stream/blocksize
    size_t global_offset=0;
    size_t device_offset=0;
    std::cerr<<"generating weights.. "<<std::endl;
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


void NetworkGenetic::allocateHostAndGPUObjects( float pMax, size_t deviceRam, size_t hostRam){
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(NetKern, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    size_t totalHost = hostRam*pMax;
    size_t totalDevice = deviceRam;
    std::cerr<<"total free device ram : "<<deviceRam<<std::endl;
    std::cerr<<"total free host ram : "<<hostRam<<std::endl;
    //each memory block is divisible by the size of an individual, the size of double, and the blocksize
    _streamSize = totalDevice/(sizeof(double)*2);

    while(_streamSize%_hostParams.array[2] || (_streamSize/_hostParams.array[2])&(_streamSize/_hostParams.array[2]-1)) // get the largest number divisible by the individual size, the threads in a block, and the size of a double
        _streamSize= _streamSize -1;

    _streambytes = _streamSize*sizeof(double);
    totalDevice = _streambytes*2;
    assert(totalDevice == _streambytes*2);
    _numOfStreams = totalHost/_streambytes; // number of streams = totalHost/streamBytes, device does not store extra weights for simplicity.
    totalHost = _streambytes*_numOfStreams;
    assert(_streambytes * _numOfStreams == totalHost);
    std::cerr<<"number of streams: "<<_numOfStreams<<std::endl;
    device_genetics.size = (totalDevice)/sizeof(double);
    host_genetics.size = totalHost/sizeof(double);
    std::cerr<<"device ram to allocate: "<<totalDevice<<std::endl;
    std::cerr<<"host ram to allocate: "<<totalHost<<std::endl;
    std::cerr<<"stream size: "<<_streamSize<<std::endl;
    CUDA_SAFE_CALL(cudaHostAlloc((void**)&host_genetics.array, totalHost, cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc((void**) &device_genetics.array, totalDevice));
    std::fill(host_genetics.array, host_genetics.array+host_genetics.size, 0);
    CUDA_SAFE_CALL(cudaMemset(device_genetics.array, 0, totalDevice));
    std::cerr<<"all allocated, moving on."<<std::endl;
    _stream.resize(_numOfStreams);
    for(int i=0; i<_numOfStreams; i++){
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream.at(i)));
        CUDA_SAFE_CALL( cudaStreamQuery(_stream.at(i)));
    }
    this->setParams();
}
bool NetworkGenetic::init(int sampleRate, int SiteNum, std::vector<double> *siteData){
    _hostParams.array[24] = sampleRate;
    _hostParams.array[23] = SiteNum;
    _siteData = siteData;
    _istraining = false;
    _hostParams.array[2] = _hostParams.array[0] + _hostParams.array[1] + 2 + 3*_hostParams.array[23]; //1*numOfSites for community mag, 1 for fitness, and 1 for age.
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
    _hostParams.array[20] = _hostParams.array[19] + _hostParams.array[10] *1; //community Magnitude offset
    _hostParams.array[21] = _hostParams.array[20] + _hostParams.array[10] *_hostParams.array[23]; // when offset.
    _hostParams.array[22] = _hostParams.array[21] + _hostParams.array[10] *_hostParams.array[23]; // howCertain offset.
    _hostParams.array[25] = _hostParams.array[22] + _hostParams.array[10] *_hostParams.array[23]; // age offset.
    std::cerr<<"allocating params memory"<<std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_deviceParams.array, _hostParams.size*sizeof(int)));
    std::cerr<<"setting params memory"<<std::endl;
    std::cerr<<"number of individuals in stream is: "<<_hostParams.array[10]<<std::endl;
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

void NetworkGenetic::reformatTraining(std::vector<int>* old_input, std::vector<double> ans, std::vector<double>* sitedata, std::vector<double>* globalquakes, double kp){ // increase the timestep and reduce resolution, takes too long.
    int trainingSize = 10;
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
                    stor[siteOffset[k]+chanOffset[j]+step] += old_input->at(k*_hostParams.array[24]*3600*3 + j*_hostParams.array[24]*3600 + step*_hostParams.array[24]*3600/trainingSize+i);
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
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(siteData, sitedata->data(), sitedata->size()*sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(globalQuakes,globalquakes->data(), globalquakes->size()*sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(Kp, &kp, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(inputData, new_input, trainingSize*_hostParams.array[23]*3*sizeof(int), 0,  cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(trainingsize, &trainingSize, sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(site_offset, siteOffset, _hostParams.array[23]*sizeof(int), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(channel_offset, chanOffset, 3*sizeof(int), 0, cudaMemcpyHostToDevice));
    delete[] siteOffset;
    delete[] chanOffset;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void NetworkGenetic::forecast(std::vector<double> *ret, int &hour, std::vector<int> *data, double &Kp, std::vector<double> *globalQuakes)
{
    //were going to normalize the inputs using v` = v-mean/stdev, so we need mean and stdev for each channel.
    double meanCh[3]{0}, stdCh[3]{0};
    int num=0;
    for(int i=0; i<3600*_hostParams.array[24]; i++){
        for(int j=0; j < _hostParams.array[23]; j++){
            for(int k=0; k<3; k++){
                meanCh[k] += data->at(3600*_hostParams.array[24]*j*3 + k*3600*_hostParams.array[24]+i);
            }
            num++;
        }
    }
    for(int k=0; k<3; k++){
        meanCh[k] = meanCh[k]/num;
        stdCh[k] = sqrt(meanCh[k]);
    }
    //input data from all sites and all channels normalized
    if(_istraining == true){
        if(hour == 50){
            cudaDeviceReset();
            exit(1);
        }
        kernelArray<double>retVec, partial_reduce_sums, dmeanCh, dstdCh;
        kernelArray<devicePair<dcon, dcon> > dConnect;
        dConnect.array = NULL;
        int regBlockSize = 512;
        int regGridSize = (_hostParams.array[10])/regBlockSize;
        retVec.size = 2160*_hostParams.array[23];
        partial_reduce_sums.size = (regGridSize);
        double *hfitnessAvg, *dfitnessAvg;
        int *hchildOffset, *dchildOffset;
        int *evoGridSize;
        int itr=0;
        for(std::vector<std::pair<hcon, hcon> >::iterator it = _connect->begin(); it != _connect->end(); ++it){
            dConnect.array[itr].first.first = it->first.first;
            dConnect.array[itr].first.second = it->first.second;
            dConnect.array[itr].second.first =it->second.first;
            dConnect.array[itr].second.second = it->second.second;
            itr++;
        }
        dConnect.size = _connect->size();
        CUDA_SAFE_CALL(cudaHostAlloc((void**)&hchildOffset, _numOfStreams*sizeof(int), cudaHostAllocWriteCombined));
        CUDA_SAFE_CALL(cudaHostAlloc((void**)&hfitnessAvg, _numOfStreams*sizeof(double), cudaHostAllocWriteCombined));
        CUDA_SAFE_CALL(cudaMalloc((void**)&evoGridSize, _numOfStreams*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dchildOffset, _numOfStreams*sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dfitnessAvg, _numOfStreams*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&retVec.array, ret->size()*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&partial_reduce_sums.array, partial_reduce_sums.size*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dmeanCh.array, 3*sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&dstdCh.array, 3*sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpy(dmeanCh.array, meanCh, 3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dstdCh.array, stdCh, 3*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(retVec.array, 0, retVec.size*sizeof(double)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(connections, dConnect.array, dConnect.size*sizeof(devicePair<dcon, dcon>), 0, cudaMemcpyHostToDevice));
        this->reformatTraining(data, _answers, _siteData,  globalQuakes, Kp);
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

            NetKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics,_deviceParams, hour, dmeanCh, dstdCh, device_offset);

            reduceFirstKern<<<regGridSize, regBlockSize, regBlockSize*sizeof(double), _stream[n]>>>(device_genetics, partial_reduce_sums, _deviceParams, device_offset);

            reduceSecondKern<<<1, 1, 0, _stream[n]>>>(partial_reduce_sums, _deviceParams, &dfitnessAvg[n]);

            for(int k=2; k<= _hostParams.array[10]; k<<= 1){
                for(int j =k>>1; j>0; j=j>>1){
                    bitonicSortKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, j, k, device_offset);
                }
            }

            normalizeKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, &dfitnessAvg[n], device_offset);

            cutoffKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams,  &dchildOffset[n], &evoGridSize[n], &dfitnessAvg[n], device_offset);

            CUDA_SAFE_CALL(cudaMemcpyAsync(&hchildOffset[n], &dchildOffset[n], sizeof(int), cudaMemcpyDeviceToHost, _stream[n]));


            evolutionKern<<<regGridSize, regBlockSize, 0, _stream[n]>>>(device_genetics, _deviceParams, &dchildOffset[n], &evoGridSize[n], seed[n], device_offset);

            CUDA_SAFE_CALL(cudaPeekAtLastError());

            CUDA_SAFE_CALL(cudaMemcpyAsync(&hfitnessAvg[n], &dfitnessAvg[n], sizeof(double), cudaMemcpyDeviceToHost, _stream[n]));

            CUDA_SAFE_CALL(cudaMemcpyAsync(&host_genetics.array[host_offset], &device_genetics.array[device_offset], _streambytes, cudaMemcpyDeviceToHost, _stream[n]));
            host_offset += _streamSize;
            device_offset += _streamSize;
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        std::cerr.precision(25);
        for(int i=0; i<100; i++){
            std::cerr<<host_genetics.array[_hostParams.array[19]+i]<<std::endl;
        }
        for(int j=0; j<_numOfStreams; j++){
            std::cerr<<"for stream #: "<<j<<" average fitness is: "<<hfitnessAvg[j]<<std::endl;
        }
        //        int ctr=0;
        //        for(int i=0; i<_hostParams.array[10]; i++){
        //            if(host_genetics.array[_hostParams.array[19] + i] >0)
        //                ctr++;
        //        }

        //        std::cerr<<"percentage %: "<<((double)ctr/(double)_hostParams.array[10])*100<<std::endl;
        int age =0, oldest=0;
        for(int i=0; i<_hostParams.array[10]; i++){
            if(host_genetics.array[_hostParams.array[25] + i] >=age){
                age = host_genetics.array[_hostParams.array[25] + i];
                oldest = i;
            }
        }
        std::cerr<<"oldest individual is: "<<oldest<<" with an age of: "<<age<<std::endl;
        std::cerr<<"with a weight #0 of "<<host_genetics.array[_hostParams.array[11] + oldest]<<std::endl;

        CUDA_SAFE_CALL(cudaFree(retVec.array));
        CUDA_SAFE_CALL(cudaFree(partial_reduce_sums.array));
        CUDA_SAFE_CALL(cudaFreeHost(hfitnessAvg));
        CUDA_SAFE_CALL(cudaFreeHost(hchildOffset));
        CUDA_SAFE_CALL(cudaFree(dfitnessAvg));
        CUDA_SAFE_CALL(cudaFree(evoGridSize));
        CUDA_SAFE_CALL(cudaFree(dchildOffset));
        CUDA_SAFE_CALL(cudaFree(dmeanCh.array));
        CUDA_SAFE_CALL(cudaFree(dstdCh.array));
        delete[] seed;
    }
    else{
        std::cerr<<"entered not training version.."<<std::endl;
        typedef std::vector<std::pair<hcon, hcon> > connectPairMatrix;
        //replace this later
        //        _best.resize(_hostParams.array[1]);
        //        for(std::vector<double>::iterator it = _best.begin(); it != _best.end(); ++it){
        //            std::srand(std::time(NULL)+*it);
        //            *it = (double)(std::rand())/(RAND_MAX);
        //        }
        std::cerr<<"example best vector has been set."<<std::endl;
        double CommunityLat = 0;
        double CommunityLon = 0;
        std::vector<double> When(_hostParams.array[23], 0);
        std::vector<double> HowCertain(_hostParams.array[23],0);
        std::vector<double> CommunityMag(_hostParams.array[23], 1); //give all sites equal mag to start, this value is [0,1]
        std::cerr<<"all output vectors created and initialized."<<std::endl;
        for(int step=0; step<3600*_hostParams.array[24]; step++){
            for(int j=0; j<_hostParams.array[23]; j++){ //sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
                CommunityLat += _siteData->at(j*2)*CommunityMag[j];
                CommunityLon += _siteData->at(j*2+1)*CommunityMag[j];
            }
            CommunityLat = CommunityLat/_hostParams.array[23];
            CommunityLon = CommunityLon/_hostParams.array[23];

            for(int j=0; j<_hostParams.array[23]; j++){ // each site is run independently of others, but shares an output from the previous step
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
                for(int k=0; k<3; k++){
                    input[k] = normalize((double)(data->at(3600*_hostParams.array[24]*j*3 + k*(3600*_hostParams.array[24])+step)), meanCh[k], stdCh[k]);

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
                for(int gate = 0; gate<_hostParams.array[15]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memgates
                        //memGateIn
                        if(it->second.first == typeMemGateIn &&it->second.second == gate && it->first.first == typeInput){ //for inputs
                            memGateIn.at(gate) += input[it->first.second]*_best[n++]; // memGateIn vect starts at 0
                        }
                        else if(it->second.first == typeMemGateIn && it->second.second == gate && it->first.first == typeHidden){//for hidden neurons
                            memGateIn.at(gate) += hidden[it->first.second]*_best[n++];
                        }
                        //memGateOut
                        else if(it->second.first == typeMemGateOut && it->second.second == gate && it->first.first == typeInput){//for inputs
                            memGateOut.at(gate) += input[it->first.second]*_best[n++];
                        }
                        else if(it->second.first == typeMemGateOut && it->second.second == gate && it->first.first == typeHidden){//for hidden neurons
                            memGateOut.at(gate) += hidden[it->first.second]*_best[n++];
                        }
                        //memGateForget
                        if(it->second.first == typeMemGateForget && it->second.second == gate && it->first.first == typeInput){//for inputs
                            memGateForget.at(gate) += input[it->first.second]*_best[n++];
                        }
                        else if(it->second.first == typeMemGateForget && it->second.second == gate && it->first.first == typeHidden){//for hidden neurons
                            memGateForget.at(gate) += hidden[it->first.second]*_best[n++];
                        }
                    }
                    memGateIn.at(gate) = ActFunc(memGateIn.at(gate));
                    memGateOut.at(gate) = ActFunc(memGateOut.at(gate));
                    memGateForget.at(gate) = ActFunc(memGateForget.at(gate));
                }
                //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
                for (int gate = 0; gate<_hostParams.array[15]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
                    if(memGateIn[gate] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->second.first == typeMemGateIn && it->second.second == gate && it->first.first == typeInput){//only pass inputs
                                mem[gate] += input[it->first.second]; // no weights attached, but the old value stored here is not removed.
                            }
                        }
                    }
                }
                for (int gate = 0; gate<_hostParams.array[16]; gate++){
                    if(memGateForget[gate] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->first.first == typeMemGateForget && it->first.second == gate && it->second.first == typeMemory){//any memory neuron that this memGateForget neuron connects to, is erased.
                                mem[it->second.second] = 0;
                            }
                        }
                    }
                }
                for (int gate = 0; gate<_hostParams.array[17]; gate++){
                    if(memGateOut.at(gate) > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->first.first == typeMemory && it->first.second == gate && it->second.first == typeHidden){ //for hidden
                                hidden[it->second.second] += mem[gate];
                            }
                            else if(it->first.first == typeMemory && it->first.second == gate && it->second.first == typeOutput){//for outputs
                                outputs[it->second.second] += mem[gate];
                            }
                        }
                    }
                }

                // hidden neuron nodes --
                for(int hid=0; hid<_hostParams.array[13]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){ // Add the inputs to the hidden neurons
                        if(it->second.first == typeHidden && it->second.second == hid && it->first.first == typeInput){ // if an input connects with this hidden neuron
                            hidden[hid] += input[it->first.second]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
                        if(it->second.first == typeHidden && it->second.second == hid && it->first.first == typeHidden){
                            hidden[hid] += hidden[it->first.second]*_best[n++];
                        }
                    }
                    hidden[hid] += 1*_best[n++]; // add bias
                    hidden[hid] = ActFunc(hidden[hid]); // then squash it.
                }
                //output nodes --

                for(int out =0; out<_hostParams.array[14]; out++){// add hidden neurons to the output nodes
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                        if(it->second.first == typeOutput && it->second.second == out && it->first.first == typeHidden){
                            outputs[out] += hidden[it->first.second]*_best[n++];
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
        float maxCertainty=0;
        float whenGuess=0;
        float guessLat=0;
        float guessLon=0;
        for(int j=0; j<_hostParams.array[23]; j++){
            if(HowCertain[j] > maxCertainty){
                maxCertainty = HowCertain[j];
                whenGuess = When[j];
                guessLat = _siteData->at(j*2);
                guessLon = _siteData->at(j*2+1);
            }
        }
        float ansLat = _siteData->at((int)_answers[0]*2);
        float ansLon = _siteData->at((int)_answers[0]*2+1);
        int whenAns = (int)_answers[1]-hour;
        double oldFit = ret->at(0);
        ret->at(0) = scoreFunc(whenGuess, whenAns, guessLat, guessLon, ansLat, ansLon, oldFit);//larger is better, negative numbers are impossible.
    }
}
