#include <network.h>
#include <getsys.h>
#include <connections.h>
#include <iostream>
#include <vector>
#include <connections.h>
#include <map>
#include <cstdlib>

using std::make_pair;
int main(int argc, char** arg){
    int numinput = 9;
    int numHidden = 3;
    int numMemory = 3; //LSTM neurons
    int numMemGateIn = 3;
    int numMemGateOut = 3;
    int numMemGateForget = 3;
    int numOutputs = 3;
    int numWeights;
    std::vector<std::pair<hcon, hcon> >connections;
    //beginning of 3 channel data inputs

    //inputs 0-2 are site sensor data (3 channels, 3 inputs per timestep)
    connections.push_back(make_pair(make_pair(typeInput, 0), make_pair(typeHidden, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 0), make_pair(typeMemGateIn, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 0), make_pair(typeMemGateOut, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 0), make_pair(typeMemGateForget, 0)));

    connections.push_back(make_pair(make_pair(typeInput, 1), make_pair(typeHidden, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 1), make_pair(typeMemGateIn, 0))); // connect input 1 to memorygateIn 0
    connections.push_back(make_pair(make_pair(typeInput, 1), make_pair(typeMemGateOut, 0)));  //connect input 1 to memory gate out 1 (all 3 data channels share a memory
    connections.push_back(make_pair(make_pair(typeInput, 1), make_pair(typeMemGateForget, 0)));

    connections.push_back(make_pair(make_pair(typeInput, 2), make_pair(typeHidden, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 2), make_pair(typeMemGateIn, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 2), make_pair(typeMemGateOut, 0)));
    connections.push_back(make_pair(make_pair(typeInput, 2), make_pair(typeMemGateForget, 0)));


    //global quake avg scalar distance is input 3
    connections.push_back(make_pair(make_pair(typeInput, 3), make_pair(typeHidden, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 3), make_pair(typeMemGateIn, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 3), make_pair(typeMemGateOut, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 3), make_pair(typeMemGateForget,1)));
    //global quake avg bearing is input 4
    connections.push_back(make_pair(make_pair(typeInput, 4), make_pair(typeHidden, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 4), make_pair(typeMemGateIn, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 4), make_pair(typeMemGateOut, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 4), make_pair(typeMemGateForget, 1)));
    //global quake avg magnitude is input 5
    connections.push_back(make_pair(make_pair(typeInput, 5), make_pair(typeHidden, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 5), make_pair(typeMemGateIn, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 5), make_pair(typeMemGateOut, 1)));
    connections.push_back(make_pair(make_pair(typeInput, 5), make_pair(typeMemGateForget, 1)));
    //planetary electromagnetic actvity index is input 6
    connections.push_back(make_pair(make_pair(typeInput, 6), make_pair(typeHidden, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 6), make_pair(typeMemGateIn, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 6), make_pair(typeMemGateOut, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 6), make_pair(typeMemGateForget, 2)));
    //community avg scalar distance is input 7
    connections.push_back(make_pair(make_pair(typeInput, 7), make_pair(typeHidden, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 7), make_pair(typeMemGateIn, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 7), make_pair(typeMemGateOut, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 7), make_pair(typeMemGateForget, 2)));
    //community avg bearing is input 8
    connections.push_back(make_pair(make_pair(typeInput, 8), make_pair(typeHidden, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 8), make_pair(typeMemGateIn, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 8), make_pair(typeMemGateOut, 2)));
    connections.push_back(make_pair(make_pair(typeInput, 8), make_pair(typeMemGateForget, 2)));

    //connect memory gate in/outs/forgets to the memory node, and vice versa.
    connections.push_back(make_pair(make_pair(typeMemGateIn, 0), make_pair(typeMemory, 0)));
    connections.push_back(make_pair(make_pair(typeMemGateOut, 0), make_pair(typeMemory, 0)));
    connections.push_back(make_pair(make_pair(typeMemGateForget, 0), make_pair(typeMemory, 0)));
    connections.push_back(make_pair(make_pair(typeMemory, 0), make_pair(typeMemGateIn, 0)));
    connections.push_back(make_pair(make_pair(typeMemory, 0), make_pair(typeMemGateOut, 0)));
    connections.push_back(make_pair(make_pair(typeMemory, 0), make_pair(typeMemGateForget, 0)));


    connections.push_back(make_pair(make_pair(typeMemGateIn, 1), make_pair(typeMemory, 1)));
    connections.push_back(make_pair(make_pair(typeMemGateOut, 1), make_pair(typeMemory, 1)));
    connections.push_back(make_pair(make_pair(typeMemGateForget, 1), make_pair(typeMemory, 1)));
    connections.push_back(make_pair(make_pair(typeMemory, 1), make_pair(typeMemGateIn, 1)));
    connections.push_back(make_pair(make_pair(typeMemory, 1), make_pair(typeMemGateOut, 1)));
    connections.push_back(make_pair(make_pair(typeMemory, 1), make_pair(typeMemGateForget, 1)));

    connections.push_back(make_pair(make_pair(typeMemGateIn, 2), make_pair(typeMemory, 2)));
    connections.push_back(make_pair(make_pair(typeMemGateOut, 2), make_pair(typeMemory, 2)));
    connections.push_back(make_pair(make_pair(typeMemGateForget, 2), make_pair(typeMemory, 2)));
    connections.push_back(make_pair(make_pair(typeMemory, 2), make_pair(typeMemGateIn, 2)));
    connections.push_back(make_pair(make_pair(typeMemory, 2), make_pair(typeMemGateOut, 2)));
    connections.push_back(make_pair(make_pair(typeMemory, 2), make_pair(typeMemGateForget, 2)));

    //connect memory neurons to hidden neurons, they don't have weights
    connections.push_back(make_pair(make_pair(typeMemory, 0), make_pair(typeHidden, 0)));
    connections.push_back(make_pair(make_pair(typeMemory, 1), make_pair(typeHidden, 1)));
    connections.push_back(make_pair(make_pair(typeMemory, 2), make_pair(typeHidden, 2)));
    //connect hidden neurons to output neurons.
    connections.push_back(make_pair(make_pair(typeHidden, 0), make_pair(typeOutput, 0)));
    connections.push_back(make_pair(make_pair(typeHidden, 0), make_pair(typeOutput, 1)));
    connections.push_back(make_pair(make_pair(typeHidden, 0), make_pair(typeOutput, 2)));

    connections.push_back(make_pair(make_pair(typeHidden, 1), make_pair(typeOutput, 0)));
    connections.push_back(make_pair(make_pair(typeHidden, 1), make_pair(typeOutput, 1)));
    connections.push_back(make_pair(make_pair(typeHidden, 1), make_pair(typeOutput, 2)));

    connections.push_back(make_pair(make_pair(typeHidden, 2), make_pair(typeOutput, 0)));
    connections.push_back(make_pair(make_pair(typeHidden, 2), make_pair(typeOutput, 1)));
    connections.push_back(make_pair(make_pair(typeHidden, 2), make_pair(typeOutput, 2)));
    numWeights = connections.size()-numMemory; //minus 3 because the memory neurons connect without weights.
    std::cerr<<"num of weights is: "<<numWeights<<std::endl;
    NetworkGenetic ConstructedNetwork(numinput, numHidden, numMemory, numMemGateIn, numMemGateOut, numMemGateForget, numOutputs, numWeights, connections);
    int sampleRate, numberOfSites, SLEN;
    std::cin>>sampleRate>>numberOfSites>>SLEN;
    std::vector<double> sitesData;

    for (int i=0; i < SLEN; i++){
        sitesData.push_back(0);
        std::cin>>sitesData.at(i);
    }
    int initRet= ConstructedNetwork.init(sampleRate, numberOfSites, &sitesData);
    std::cout<<initRet<<std::endl;
    int doTraining;
    std::cin>>doTraining;
    if (doTraining == 1)
    {
        int gtf_site, gtf_hour;
        double gtf_lat, gtf_long, gtf_mag, gtf_dist;
        std::cin>>gtf_site>>gtf_hour>>gtf_lat>>gtf_long>>gtf_mag>>gtf_dist;

        if(ConstructedNetwork.checkForWeights("/weights.bin"))
            ConstructedNetwork.generateWeights();
        else{
            ConstructedNetwork.allocateHostAndGPUObjects(0.75, GetDeviceRamInBytes(), GetHostRamInBytes());
            ConstructedNetwork.generateWeights();
        }
        ConstructedNetwork.doingTraining(gtf_site, gtf_hour, gtf_lat, gtf_long, gtf_mag, gtf_dist);
    }
    while(1)
    {
        int DLEN, QLEN;
        int hour;
        double Kp;
        std::vector<int> data ;
        std::vector<double> globalQuakes(5, 0);
        std::vector<double> tmpQuakes;
        std::vector<double> retM(2160*numberOfSites, 0);


        std::cin>>hour;
        if(hour== -1)
            break;
        std::cin>>DLEN;
        for(int i=0; i<DLEN; i++){
            data.push_back(0);
            std::cin>>data.at(i);
            if(data.at(i) == -1){
                data.at(i) = data.at(i-1);
            }
        }
        std::cin>>Kp;
        std::cin>>QLEN;

        for(int i=0; i<QLEN; i++){
            tmpQuakes.push_back(0.0);
            std::cin>>tmpQuakes.at(i);
        }
        int accVal=0;
        for (int i=0; i<QLEN/5; i++){
            for(int k=0; k<4; k++){//don't start at 0 because 0 is time.
                globalQuakes.at(k) += tmpQuakes.at(i*5+k);
                accVal++;
            }
        }
        for(int k=0; k<4; k++){
            if(globalQuakes.at(k) !=0)
                globalQuakes.at(k) = globalQuakes.at(k)/accVal; // push the hourly average into _DGQuakes for all parameters.
        }
        ConstructedNetwork.forecast(&retM, hour, &data, Kp, &globalQuakes);
        std::cout<<retM.size()<<std::endl;
        for(unsigned int i=0; i<retM.size(); i++){
            std::cout<<retM.at(i)<<std::endl;
        }
        std::cout.flush();
    }
    if(doTraining == 1)
        ConstructedNetwork.storeWeights("/weights.bin");
    return 0;
}
