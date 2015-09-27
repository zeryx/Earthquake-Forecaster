#ifndef NETWORK_H
#define NETWORK_H
#include <dataarray.h>
#include <connections.h>
#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include <utility>
#include <getsys.h>

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do{cudaError_t err = call; if (cudaSuccess != err) {fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",__FILE__, __LINE__, cudaGetErrorString(err) ); cudaDeviceReset(); exit(EXIT_FAILURE);}} while (0)
#endif

class  NetworkGenetic{
public:
    NetworkGenetic();

    void generateWeights(); //initializes _data array and fills with random numbers

    void allocateHostAndGPUObjects(size_t deviceRam = GetDeviceRamInBytes()*0.85, size_t hostRam = GetHostRamInBytes()*0.85);

    void trainForecast(std::vector<double> *ret, int &hour, std::vector<int> &data, double &Kp, std::vector<double> &globalQuakes,
                       Order *connections, std::vector<double> &answers, std::vector<double> &siteData);


    void training();

    void challengeForecast(std::vector<double> *ret, int &hour, std::vector<int> &data, double &K, std::vector<double> &globalQuakes, Order *connections, std::vector<double> &siteData);

    void reformatTraining(std::vector<int>& old_input, std::vector<double> &ans, std::vector<double> &sitedata, std::vector<double>& globalquakes, double& kp);


    void setParams(int num, int val);

    void confDeviceParams();

    void confNetParams(const int &numInNeurons, const int &numHiddenNeurons, const int &numMemoryNeurons, const int &numMemoryIn,
                         const int &numMemoryOut, const int &numMemoryForget,  const int &numOutNeurons);

    void confOrder(const int &numOrders, const int &numWeights);

    void confTestParams(const int &numOfSites, const int &sampleRate);

    bool loadFromFile(std::ifstream &stream);

    void saveToFile(std::ofstream &stream);


        ~NetworkGenetic();
private:
    int _numOfStreams;
    long int _streambytes;
    long int _streamSize;
    kernelArray<double> device_genetics;
    kernelArray<double> host_genetics;
    std::vector<double> _best;
    kernelArray<int>_hostParams;
    kernelArray<int>_deviceParams;
    std::vector<cudaStream_t> _stream;
};


#endif
