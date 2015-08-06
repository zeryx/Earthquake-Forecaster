#include "network.h"
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <utility>
#include <ctime>
#include <thrust/host_vector.h>

//macros
//cuda error message handling
#define CUDA_SAFE_CALL(call)                                          \
    do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
    __FILE__, __LINE__, cudaGetErrorString(err) );       \
    exit(EXIT_FAILURE);                                           \
    }                                                                 \
    } while (0)

//neural functions
__host__ __device__ inline double sind(double x)
{
    return sin(x * M_PI / 180);
}

__host__ __device__ inline double cosd(double x)
{
    return cos(x * M_PI / 180);
}
__host__ __device__ inline double distCalc(double lat1, double lon1, double lat2, double lon2){
    double earthRad = 6371.01;
    double deltalon = abs(lon1 - lon2);
    if(deltalon > 180)
        deltalon = 360 - deltalon;
    return earthRad * atan2( sqrt( pow( cosd(lat1) * sind(deltalon), 2) +
                                   pow( cosd(lat2) * sind(lat1) - sind(lat2) * cosd(lat1) * cosd(deltalon), 2) ),
                             sind(lat2) * sind(lat1) + cosd(lat2) * cosd(lat1) * cosd(deltalon));
}

__host__ __device__ inline double bearingCalc(double lat1, double lon1, double lat2, double lon2){
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

__host__ __device__ inline double ActFunc(double x){
    return tanh(x);
}

template <typename T>
__global__ void genWeights( dataArray<T> ref, long in, int nRegWeights, int indLength){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::minstd_rand0 randEng;
    randEng.seed(idx);
    long seed = idx+ref.size*in;
    for(int i=0; i<nRegWeights; i++){
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref.array[idx*indLength + i] = uniDist(randEng);
    }
}

__global__ void Net(dataArray<double> weights, dataArray<int> params, dataArray<double> globalQuakes,
                    dataArray<int> inputVal, dataArray<double> siteData,
                    dataArray<double> answers, dataArray<double> returnVal,
                    dataArray<thrust::pair<int, int> > connections, double Kp, int sampleRate,int numOfSites, int hour){



    int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread, calculate a individuals weight.
    int ind = idx*params.array[7];
    typedef thrust::device_ptr<thrust::pair<int, int> > connectPairMatrix;
    double CommunityLat = 0;
    double CommunityLon = 0;
    double *When = (double*)malloc(numOfSites*sizeof(double));
    double *HowCertain = (double*)malloc(numOfSites*sizeof(double));
    double *CommunityMag = (double*)malloc(numOfSites*sizeof(double)); //give all sites equal mag to start, this value is [0,1]

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
            //replace these with real connections, num of inputs, and num of hidden & memory neurons (mem neurons probably accurate)
            int *input = (int*)malloc(params.array[3]*sizeof(int)); // number of inputs is 9.
            double *hidden = (double*)malloc(params.array[4]*sizeof(double)); // for practice sake, lets say each input has its own neuron (might be true!)
            double *mem = (double*)malloc(params.array[5]*sizeof(double)); // stores the input if gate is high
            double *memGateIn = (double*)malloc(params.array[5]*sizeof(double)); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
            double *memGateOut = (double*)malloc(params.array[5]*sizeof(double));
            double *memGateForget = (double*)malloc(params.array[5]*sizeof(double));
            double *outputs = (double*)malloc(params.array[6]*sizeof(double)); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0;
            int startOfInput = 0;
            int startOfHidden = startOfInput +params.array[3];
            int startOfMem = startOfHidden + params.array[4];
            int startOfMemGateIn = startOfMem + params.array[5];
            int startOfMemGateOut = startOfMemGateIn + params.array[5];
            int startOfMemGateForget = startOfMemGateOut + params.array[5];
            int startOfOutput = startOfMemGateForget + params.array[5];
            input[0] = 1/inputVal.array[(3600*sampleRate*j*3 + 1*(3600*sampleRate)+step)];//channel 1
            input[1] = 1/inputVal.array[(3600*sampleRate*j*3 + 2*(3600*sampleRate)+step)];//channel 2
            input[2] = 1/inputVal.array[(3600*sampleRate*j*3 + 3*(3600*sampleRate)+step)];//channel 3
            input[3] = 1/GQuakeAvgdist;
            input[4] = 1/GQuakeAvgBearing;
            input[5] = 1/GQuakeAvgMag;
            input[6] = 1/Kp;
            input[7] = 1/CommunityDist;
            input[8] = 1/CommunityBearing;
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


            When[j] += 1/outputs[0]; //return when back to an integer value (adjust to fit within boundaries)
            HowCertain[j] += outputs[1];
            CommunityMag[j] =  outputs[2]; // set the next sets communityMag = output #3.
        }
    }
    for(int j=0; j<numOfSites; j++){ // each site has its own when and howcertain vector
        When[j] = When[j]/3600*sampleRate;
        HowCertain[j] = HowCertain[j]/3600*sampleRate;
    }
    //all done, lets output the return matrix.
    //since right now were using a point value for when & how certain (only one output per site),
    //we're going to approximate using a normal distribution around when with a sigma of howCertain, over the whole array from T=currentHour [T, 2160]
    for(int h=hour; h<2160; h++){
        for(int j=0; j<numOfSites; j++){
            returnVal.array[h*numOfSites+j] = 1/(1/HowCertain[j]*sqrt(2*M_PI))*exp(-pow(h-When[j], 2)/(2*pow(1/HowCertain[j], 2))); // normal distribution with a mu of When and a sigma of 1/HowCertain
        }
    }
}



NetworkGenetic::NetworkGenetic(const int &numInputNodes, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, std::vector< thrust::pair<int, int> >&connections){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInputNodes + numHiddenNeurons + numMemoryNeurons + numOutNeurons;
    _NNParams[2] = numInputNodes + numHiddenNeurons + numOutNeurons;
    _NNParams[3] = numInputNodes;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = numInputNodes + numHiddenNeurons + numMemoryNeurons + numOutNeurons + 1 + 1; //1 for fitness, 1 for community output composite vector
    _connect = &connections;
}

void NetworkGenetic::initializeWeights(){
    int blocksPerGrid; //the blocksize defined by the configurator
    int threadsblock = 512; // the actual grid size needed
    int seedItr = 0;
    //    do{
    _NNParams[8] = _memVirtualizer._DGenetics.size()/(_NNParams[7]); // number of individuals on device.
    long seed = std::clock() + std::clock()*seedItr++;
    blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
    genWeights<double><<<blocksPerGrid, threadsblock>>>(_memVirtualizer.genetics(), seed, _NNParams[2], _NNParams[8]);
    cudaDeviceSynchronize();
    //    }while(_memVirtualizer.GeneticsPushToHost(&_genetics));
    //    _NNParams[9] = _genetics.size/(_NNParams[8]); // number of individuals on device.
    //    long seed = std::clock() + std::clock()*seedItr++;
    //    blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
    //    genWeights<double><<< blocksPerGrid, threadsblock>>>(_genetics, seed, _NNParams[2], _NNParams[8]);
    //    cudaDeviceSynchronize();
}


void NetworkGenetic::allocateHostAndGPUObjects(float pMaxHost, float pMaxDevice){
    _memVirtualizer.memoryAlloc(_NNParams[8], pMaxHost, pMaxDevice);

}
bool NetworkGenetic::init(int sampleRate, int SiteNum, std::vector<double> siteData){
    _sampleRate = sampleRate;
    _numofSites = SiteNum;
    _siteData.resize(siteData.size());
    try{thrust::copy(siteData.begin(), siteData.end(), _siteData.begin());}
    catch(thrust::system_error &e){
        std::cerr<<"Error resizing vector Element: "<<e.what()<<std::endl;
        return false;
    }
    catch(std::bad_alloc &e){
        std::cerr<<"Ran out of space due to : "<<"host"<<std::endl;
        std::cerr<<e.what()<<std::endl;
        return false;
    }
    _istraining = false;
    return true;
}

bool NetworkGenetic::checkForWeights(std::string filepath){
    std::ifstream weightFile;
    weightFile.open(filepath.c_str(), std::ios_base::in);
    if(weightFile){
        _memVirtualizer.initFromStream(weightFile);
        return true;
    }
    else
        return false;
}

void NetworkGenetic::doingTraining(int site, int hour, double lat, double lon, double mag, double dist){
    _answers.push_back(site);
    _answers.push_back(hour);
    _answers.push_back(lat);
    _answers.push_back(lon);
    _answers.push_back(mag);
    _answers.push_back(dist);
    _istraining = true;
}

void NetworkGenetic::storeWeights(std::string filepath){
    _memVirtualizer.pushToStream(filepath);
}

void NetworkGenetic::forecast(double *ret, int &hour, std::vector<int> *data, double &Kp, std::vector<double> *globalQuakes)
{
    std::cerr<<"entered forecast"<<std::endl;
    if(_istraining){
        thrust::device_vector<double> retVec(2160*_numofSites, 0);
        thrust::device_vector<int> input(data->size());
        thrust::device_vector<double>gQuakeAvg(globalQuakes->size());
        thrust::device_vector<thrust::pair<int, int> > dConnect(_connect->size());
        thrust::copy(_connect->begin(), _connect->end(), dConnect.begin());
        thrust::copy(data->begin(), data->end(), input.begin());
        thrust::copy(globalQuakes->begin(), globalQuakes->end(), gQuakeAvg.begin());

        int blocksPerGrid; //the blocksize defined by the configurator
        int threadsblock = 512; // the actual grid size needed

        _NNParams[8] = _memVirtualizer._DGenetics.size()/(_NNParams[7]);
        blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
        Net<<<blocksPerGrid, threadsblock>>>(_memVirtualizer.genetics(),
                                             convertToKernel(_NNParams),
                                             convertToKernel(gQuakeAvg),
                                             convertToKernel(input),
                                             convertToKernel(_siteData),
                                             convertToKernel(_answers),
                                             convertToKernel(retVec),
                                             convertToKernel(dConnect),
                                             Kp,
                                             _sampleRate,
                                             _numofSites,
                                             hour);
        cudaDeviceSynchronize();
        thrust::copy(retVec.begin(), retVec.end(), ret);
    }
    else{
        std::cerr<<"entered not training version.."<<std::endl;
        typedef std::vector<thrust::pair<int, int> > connectPairMatrix;
        //replace this later
        _best.resize(45);
        for(std::vector<double>::iterator it = _best.begin(); it != _best.end(); ++it){
            *it = 1/rand();
        }
        std::cerr<<"example best vector has been set."<<std::endl;
        double CommunityLat = 0;
        double CommunityLon = 0;
        std::vector<double> When(_numofSites);
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
                //                double avgDepthGQuake = globalQuakes->at(2); don't think I care about depth that much.
                double GQuakeAvgMag = globalQuakes->at(3);
                double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
                double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
                std::vector<int> input;
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
                int n =0;
                int startOfInput = 0;
                int startOfHidden = startOfInput +_NNParams[3];
                int startOfMem = startOfHidden + _NNParams[4];
                int startOfMemGateIn = startOfMem + _NNParams[5];
                int startOfMemGateOut = startOfMemGateIn + _NNParams[5];
                int startOfMemGateForget = startOfMemGateOut + _NNParams[5];
                int startOfOutput = startOfMemGateForget + _NNParams[5];
                input[0] = 1/data->at(3600*_sampleRate*j*3 + 1*(3600*_sampleRate)+step);
                input[1] = 1/data->at(3600*_sampleRate*j*3 + 2*(3600*_sampleRate)+step);
                input[2] = 1/data->at(3600*_sampleRate*j*3 + 3*(3600*_sampleRate)+step);
                input[3] = 1/GQuakeAvgdist;
                input[4] = 1/GQuakeAvgBearing;
                input[5] = 1/GQuakeAvgMag;
                input[6] = 1/Kp;
                input[7] = 1/CommunityDist;
                input[8] = 1/CommunityBearing;
                //lets reset all neuron values for this new timestep (except memory neurons)
                for(int gate=0; gate<_NNParams[5]; gate++){
                    memGateIn[gate] = 0;
                    memGateOut[gate] = 0;
                    memGateForget[gate] = 0;
                }
                for(int hid=0; hid<_NNParams[4]; hid++){
                    hidden[hid] = 0;
                }
                for(int out=0; out<_NNParams[6]; out++){
                    outputs[out] = 0;
                }

                //now that everything that should be zeroed is zeroed, lets start the network.
                //mem gates & LSTM nodes --
                for(int gate = 0; gate<_NNParams[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateIn
                        if(it->second == gate+startOfMemGateIn && it->second < startOfHidden){ //for inputs
                            memGateIn[gate] += input[it->first-startOfInput]*_best[n++]; // memGateIn vect starts at 0
                        }
                        else if(it->second == gate+startOfMemGateIn && it->second >startOfHidden && it->second <startOfMem){//for hidden neurons
                            memGateIn[gate] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for memGateOut
                        if(it->second == gate+startOfMemGateOut && it->second < startOfHidden){//for inputs
                            memGateOut[gate] += input[it->first-startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateOut && it->second >startOfHidden && it->second <startOfMem){//for hidden neurons
                            memGateOut[gate] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//for  memGateForget
                        if(it->second == gate+startOfMemGateForget && it->second < startOfHidden){//for inputs
                            memGateForget[gate] += input[it->first - startOfInput]*_best[n++];
                        }
                        else if(it->second == gate+startOfMemGateForget && it->second >startOfHidden && it->second <startOfMem){//for hidden neurons
                            memGateForget[gate] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    memGateIn[gate] = ActFunc(memGateIn[gate]);
                    memGateOut[gate] = ActFunc(memGateOut[gate]);
                    memGateForget[gate] = ActFunc(memGateForget[gate]);
                }
                //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
                for (int gate = 0; gate<_NNParams[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
                    if(memGateIn[gate] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->second == gate+startOfMemGateIn && it->first < gate+startOfHidden){//only pass inputs
                                mem[gate] += input[it->first-startOfInput]; // no weights attached, but the old value stored here is not removed.
                            }
                        }
                    }
                    if(memGateForget[gate] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
                        mem[gate] = 0;
                    }
                    //if memGateForget fires, then memGateOut will output nothing.
                    if(memGateOut[gate] > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
                        for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){
                            if(it->first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
                                hidden[it->second-startOfHidden] += mem[gate];
                            }
                        }
                    }
                }

                // hidden neuron nodes --
                for(int hid=0; hid<_NNParams[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){ // Add the inputs to the hidden neurons
                        if(it->second == hid+startOfHidden && it->first < startOfHidden && it->first >= startOfInput){ // if an input connects with this hidden neuron
                            hidden[hid] += input[it->first]*_best[n++];
                        }
                    }
                    for(connectPairMatrix::iterator it = _connect->begin(); it!= _connect->end(); ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
                        if(it->second == hid+startOfHidden && it->first < startOfMem && it->first >= startOfHidden){
                            hidden[hid] += hidden[it->first-startOfHidden]*_best[n++];
                        }
                    }
                    hidden[hid] += 1*_best[n++]; // add bias
                    hidden[hid] = ActFunc(hidden[hid]); // then squash it.
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


                When[j] += 1/outputs[0]; //return when back to an integer value (adjust to fit within boundaries)
                HowCertain[j] += outputs[1];
                CommunityMag[j] =  outputs[2]; // set the next sets communityMag = output #3.
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
                ret[h*_numofSites+j] = 1/(1/HowCertain[j]*sqrt(2*M_PI))*exp(-pow(h-When[j], 2)/(2*pow(1/HowCertain[j], 2))); // normal distribution with a mu of When and a sigma of 1/HowCertain
            }
        }
    }
}
