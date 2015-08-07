#include "network.h"
#include <iostream>
#include <utility>
#include <thrust/pair.h>

using std::make_pair;
int main(int argc, char** arg){
    int inputs = 9;
    int hidden = 3;
    int memory = 3; //LSTM neurons
    int memGateIn = 3;
    int MemGateOut = 3;
    int MemGateForget = 3;
    int outputs = 3;
    int numWeights;
    std::vector< thrust::pair<int, int> >connections; //each vector starts at 0, and then is < than the num of each val
    connections.push_back(make_pair(0, inputs+0));
    connections.push_back(make_pair(0, inputs+hidden+memory+0)); // connect input 1 to memorygateIn 1
    connections.push_back(make_pair(0, inputs+hidden+memory+memGateIn+0)); // connect input 1 to memory gateoutput 1
    connections.push_back(make_pair(0, inputs+hidden+memory+memGateIn+MemGateOut+0));
    connections.push_back(make_pair(1, inputs+0));
    connections.push_back(make_pair(1, inputs+hidden+memory+0));
    connections.push_back(make_pair(1, inputs+hidden+memory+memGateIn+0)); //connect input 2 to memory gate out 1 (all 3 data channels share a memory
    connections.push_back(make_pair(1, inputs+hidden+memory+memGateIn+MemGateOut+0));
    connections.push_back(make_pair(2, inputs+0));
    connections.push_back(make_pair(2, inputs+hidden+memory+0));
    connections.push_back(make_pair(2, inputs+hidden+memory+memGateIn+0));
    connections.push_back(make_pair(2, inputs+hidden+memory+memGateIn+MemGateOut+0));
    connections.push_back(make_pair(3, inputs+1));
    connections.push_back(make_pair(3, inputs+hidden+memory+1));
    connections.push_back(make_pair(3, inputs+hidden+memory+memGateIn+1));
    connections.push_back(make_pair(3, inputs+hidden+memory+memGateIn+MemGateOut+1));
    connections.push_back(make_pair(4, inputs+1));
    connections.push_back(make_pair(4, inputs+hidden+memory+1));
    connections.push_back(make_pair(4, inputs+hidden+memory+memGateIn+1));
    connections.push_back(make_pair(4, inputs+hidden+memory+memGateIn+MemGateOut+1));
    connections.push_back(make_pair(5, inputs+1));
    connections.push_back(make_pair(5, inputs+hidden+memory+1));
    connections.push_back(make_pair(5, inputs+hidden+memory+memGateIn+1));
    connections.push_back(make_pair(5, inputs+hidden+memory+memGateIn+MemGateOut+1));
    connections.push_back(make_pair(6, inputs+2));
    connections.push_back(make_pair(6, inputs+hidden+memory+2));
    connections.push_back(make_pair(6, inputs+hidden+memory+memGateIn+2));
    connections.push_back(make_pair(6, inputs+hidden+memory+memGateIn+MemGateOut+2));
    connections.push_back(make_pair(7, inputs+2));
    connections.push_back(make_pair(7, inputs+hidden+memory+2));
    connections.push_back(make_pair(7, inputs+hidden+memory+memGateIn+2));
    connections.push_back(make_pair(7, inputs+hidden+memory+memGateIn+MemGateOut+2));
    connections.push_back(make_pair(8, inputs+2));
    connections.push_back(make_pair(8, inputs+hidden+memory+2));
    connections.push_back(make_pair(8, inputs+hidden+memory+memGateIn+2));
    connections.push_back(make_pair(8, inputs+hidden+memory+memGateIn+MemGateOut+2));
    //connect memory gate in/outs/forgets to the memory node.
    connections.push_back(make_pair(inputs+hidden+memory+0, inputs+hidden+0));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+0, inputs+hidden+0));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+MemGateOut+0, inputs+hidden+0));
    connections.push_back(make_pair(inputs+hidden+memory+1, inputs+hidden+1));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+1, inputs+hidden+1));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+MemGateOut+1, inputs+hidden+1));
    connections.push_back(make_pair(inputs+hidden+memory+2, inputs+hidden+2));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+2, inputs+hidden+2));
    connections.push_back(make_pair(inputs+hidden+memory+memGateIn+MemGateOut+2, inputs+hidden+2));
    //connect memory neurons to hidden neurons, they don't have weights
    connections.push_back(make_pair(inputs+hidden+0, inputs+0));
    connections.push_back(make_pair(inputs+hidden+1, inputs+1));
    connections.push_back(make_pair( inputs+hidden+2, inputs+2));
    //connect hidden neurons to output neurons.
    connections.push_back(make_pair(inputs+0, inputs+hidden+memGateIn+MemGateOut+MemGateForget+0));
    connections.push_back(make_pair(inputs+0, inputs+hidden+memGateIn+MemGateOut+MemGateForget+1));
    connections.push_back(make_pair(inputs+0, inputs+hidden+memGateIn+MemGateOut+MemGateForget+2));
    connections.push_back(make_pair(inputs+1, inputs+hidden+memGateIn+MemGateOut+MemGateForget+0));
    connections.push_back(make_pair(inputs+1, inputs+hidden+memGateIn+MemGateOut+MemGateForget+1));
    connections.push_back(make_pair(inputs+1, inputs+hidden+memGateIn+MemGateOut+MemGateForget+2));
    connections.push_back(make_pair(inputs+2, inputs+hidden+memGateIn+MemGateOut+MemGateForget+0));
    connections.push_back(make_pair(inputs+2, inputs+hidden+memGateIn+MemGateOut+MemGateForget+1));
    connections.push_back(make_pair(inputs+2, inputs+hidden+memGateIn+MemGateOut+MemGateForget+2));
    numWeights = connections.size()-3; //minus 3 because the memory neurons connect without weights.
    NetworkGenetic ConstructedNetwork(inputs, hidden, memory, outputs,numWeights, connections);

    int sampleRate, numberOfSites, SLEN;
    std::cin>>sampleRate>>numberOfSites>>SLEN;

    std::vector<double> sitesData;

    for (int i=0; i < SLEN; i++){
        sitesData.push_back(0);
        std::cin>>sitesData.at(i);
    }
    int initRet= ConstructedNetwork.init(sampleRate, numberOfSites, sitesData);
    std::cout<<initRet<<std::endl;
    int doTraining;
    std::cin>>doTraining;
    if (doTraining == 1)
    {
        std::cerr<<"looks like were doing training"<<std::endl;
        int gtf_site, gtf_hour;
        double gtf_lat, gtf_long, gtf_mag, gtf_dist;
        std::cin>>gtf_site>>gtf_hour>>gtf_lat>>gtf_long>>gtf_mag>>gtf_dist;
        ConstructedNetwork.allocateHostAndGPUObjects(0.50, 0.75);
        ConstructedNetwork.doingTraining(gtf_site, gtf_hour, gtf_lat, gtf_long, gtf_mag, gtf_dist);
        std::cerr<<"lets allocate GPU and host objects"<<std::endl;
        std::cerr<<"weights initialized, setting training"<<std::endl;
        std::cerr<<"checking for weightfile"<<std::endl;
        if(!ConstructedNetwork.checkForWeights("/weights.bin"))
            ConstructedNetwork.initializeWeights();
    }
    while(1)
    {
        int DLEN, QLEN;
        int hour;
        double Kp;
        std::vector<int> data;
        std::vector<double> globalQuakes(5);
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
        std::cerr<<"recieved all input data"<<std::endl;
        std::cin>>Kp;
        std::cin>>QLEN;
        std::vector<double> tmpQuakes;
        for(int i=0; i<QLEN; i++){
            tmpQuakes.push_back(0.0);
            std::cin>>tmpQuakes.at(i);
        }
        std::cerr<<"recieved all global quakes data"<<std::endl;
        int accVal=0;
        globalQuakes[5] = hour;
        for (int i=0; i<QLEN; i++){
            for(int k=0; k<4; k++){//don't start at 0 because 0 is time.
                globalQuakes[k] += tmpQuakes[i*5+k];
                accVal++;
            }
        }
        for(int k=1; k<4; k++){
            if(globalQuakes[k] !=0)
                globalQuakes[k] = globalQuakes[k]/accVal; // push the hourly average into _DGQuakes for all parameters.
        }
        std::cerr<<"about to call forecast.."<<std::endl;
        double retM[2160*numberOfSites];
        ConstructedNetwork.forecast(retM, hour, &data, Kp, &globalQuakes);
        std::cerr<<"forecast returned."<<std::endl;
        int retSize = numberOfSites;
        std::cout<<retSize<<std::endl;
        for(int i=0; i<2160*numberOfSites; i++){
            std::cout<<retM[i]<<std::endl;
        }
        std::cerr<<"ret is returned to stream."<<std::endl;
        std::cout.flush();
    }
    if(doTraining == 1)
        ConstructedNetwork.storeWeights("/weights.bin");
}
