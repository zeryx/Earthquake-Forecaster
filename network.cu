#include "network.h"
#include "dataarray.h"
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <ctime>
#include <cmath>
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


template <typename T>
__global__ void genWeights( dataArray<T> ref, long in, int nRegWeights, int indLength){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    thrust::minstd_rand0 randEng;
    randEng.seed(idx);
    long seed = idx+ref._size*in;
    for(int i=0; i<nRegWeights; i++){
        thrust::uniform_real_distribution<double> uniDist(0,1);
        randEng.discard(seed);
        ref._array[idx*indLength + i] = uniDist(randEng);
    }
}

__global__ void Net(dataArray<double> weights, dataArray<int> params,
                    dataArray<int> input, dataArray<double> siteData,
                    dataArray<double> answers, dataArray<double> returnVal,
                    dataArray<int> connections, double Kp, int sampleRate, int numOfSites){



}


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
    brng = fmod((brng + 360), 360);
    brng = 360 - brng;

    return brng;
}

__host__ __device__ inline double ActFunc(double x){
    return tanh(x);
}


NetworkGenetic::NetworkGenetic(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons,
                               const int &numOutNeurons, const int &numHiddenLayers,
                               const std::vector<int>&connections){
    this->_NNParams.resize(15, 0); // room to grow
    _NNParams[1] = numInNeurons + numHiddenNeurons + numMemoryNeurons + numOutNeurons;
    _NNParams[2] = numInNeurons + numHiddenNeurons + numOutNeurons;
    _NNParams[3] = numInNeurons;
    _NNParams[4] = numHiddenNeurons;
    _NNParams[5] = numMemoryNeurons;
    _NNParams[6] = numOutNeurons;
    _NNParams[7] = numHiddenLayers;
    _NNParams[8] = numInNeurons + numHiddenLayers + numMemoryNeurons + numOutNeurons + 1 + 1; //1 for fitness, 1 for community output composite vector
    _connections.resize(connections.size());
    thrust::copy(connections.begin(), connections.end(), _connections.begin());
}

void NetworkGenetic::initializeWeights(){
    int blocksPerGrid; //the blocksize defined by the configurator
    int threadsblock = 512; // the actual grid size needed
    int seedItr = 0;
    //    do{
    _NNParams[9] = _memVirtualizer._DGenetics.size()/(_NNParams[8]); // number of individuals on device.
    long seed = std::clock() + std::clock()*seedItr++;
    blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
    genWeights<double><<<blocksPerGrid, threadsblock>>>(_memVirtualizer.genetics(), seed, _NNParams[2], _NNParams[8]);
    cudaDeviceSynchronize();
    //    }while(_memVirtualizer.GeneticsPushToHost(&_genetics));
    //    _NNParams[9] = _genetics._size/(_NNParams[8]); // number of individuals on device.
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

double* NetworkGenetic::forecast(int &hour, std::vector<int> *data, double &Kp, std::vector<double> *globalQuakes)
{
    double*  ret =(double*)calloc(2160*_numofSites, sizeof(double));

    if(_istraining){
        thrust::device_vector<double> retVec(2160*_numofSites, 0);
        thrust::device_vector<int> input(data->size());
        thrust::copy(data->begin(), data->end(), input.begin());

        int blocksPerGrid; //the blocksize defined by the configurator
        int threadsblock = 512; // the actual grid size needed

        _NNParams[9] = _memVirtualizer._DGenetics.size()/(_NNParams[8]);
        blocksPerGrid=(_NNParams[9]+threadsblock-1)/threadsblock;
        Net<<<blocksPerGrid, threadsblock>>>(_memVirtualizer.genetics(),
                                             convertToKernel(_NNParams),
                                             convertToKernel(input),
                                             convertToKernel(_siteData),
                                             convertToKernel(_answers),
                                             convertToKernel(retVec),
                                             convertToKernel(_connections),
                                             Kp, _sampleRate, _numofSites);
        cudaDeviceSynchronize();
        thrust::copy(retVec.begin(), retVec.end(), ret);
        return ret;
    }
    else{
        double CommunityLat = 0;
        double CommunityLon = 0;
        std::vector<double> When(_numofSites,0);
        std::vector<double> HowCertain(_numofSites,0);
        std::vector<double>CommunityMag(_numofSites, 1); //give all sites equal mag to start, this value is [0,1]


        for(int step=0; step<3600*_sampleRate; step++){

            for(int j=0; j<_numofSites; j++){ //sitesWeighted Lat/Lon values are determined based on all previous sites mag output value.
                CommunityLat += _siteData[j*2]*CommunityMag[j];
                CommunityLon += _siteData[j*2+1]*CommunityMag[j];
            }
            CommunityLat = CommunityLat/_numofSites;
            CommunityLon = CommunityLon/_numofSites;

            for(int j=0; j<_numofSites; j++){ // each site is run independently of others, but shares an output from the previous step
                double latSite = _siteData[j*2];
                double lonSite = _siteData[j*2+1];
                double avgLatGQuake = globalQuakes->at(0);
                double avgLonGQuake = globalQuakes->at(1);
                //                double avgDepthGQuake = globalQuakes->at(2); don't think I care about depth that much.
                double avgMagGQuake = globalQuakes->at(3);
                double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
                double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
                double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
                std::vector<double> input, hiddenN1, outputs, memN, memGateOut, memGateIn;
                //replace these with real connections, num of inputs, and num of hidden & memory neurons (mem neurons probably accurate)
                input.resize(9); // number of inputs is 9.
                hiddenN1.resize(9, 0); // for practice sake, lets say each input has its own neuron (might be true!)
                memN.resize(9, 0.0); // stores the input if gate is high
                memGateOut.resize(9, 0.0); //connects to the input layer and the memN associated with input, if 1 it sends up stream and deletes, if low it keeps.
                memGateIn.resize(9, 0.0);
                outputs.resize(3, 0.0); /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                    1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
                int n =0;
                input[0] = 1/data->at(3600*_sampleRate*j*3 + 1*(3600*_sampleRate)+step);
                input[1] = 1/data->at(3600*_sampleRate*j*3 + 2*(3600*_sampleRate)+step);
                input[2] = 1/data->at(3600*_sampleRate*j*3 + 3*(3600*_sampleRate)+step);
                input[3] = 1/GQuakeAvgdist;
                input[4] = 1/GQuakeAvgBearing;
                input[5] = 1/avgMagGQuake;
                input[6] = 1/Kp;
                input[7] = 1/CommunityDist;
                input[8] = 1/CommunityBearing;
                //for layer 1 (assuming 1 neuron for each input, will adjust with a connections matrix later
                for(int in=0; in<9; in++){ // Add the inputs to the hidden neurons
                    hiddenN1[in] += input[in]*_best[n++];
                }
                for(int in=0; in<9; in++){
                    memGateIn[in] = ActFunc(input[in]*_best[n++]);
                    if(memGateIn[in] > 0.5){ // if memGateIn >0.5, accept a new input and throw out the old val, if less, keep old val.
                        memN[in] = input[in];
                    }
                    memGateOut[in] = ActFunc(input[in]*_best[n++]);
                    if(memGateOut[in] > 0.5){ // if memGateOut >0.5, pass the memory val to the hiddenNeuron upstream and remove the mem val.
                        hiddenN1[in] += memN[0]*_best[n++];
                        memN[in] = 0;
                    }
                }
                for(int in=0; in<9; in++){
                    hiddenN1[in] += 1*_best[n++]; // add the bias neuron + weight to each hidden neuron in lvl 1.
                    hiddenN1[in] = ActFunc(hiddenN1[in]);
                }
                //now for the output layer, all hidden neurons connect with both outputs.
                for(int out=0; out<3; out++){ // for each output neuron
                    for(int in=0; in<9; in++){
                        outputs[out] += hiddenN1[in]*_best[n++]; // add all the hidden neurons
                    }
                    outputs[out] += 1*_best[n++]; // then add the bias
                    outputs[out] = ActFunc(outputs[out]); // calculate the activation val between 0, 1, normalize for whatever value its supposed to represent.
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
    return ret;
}
