#include "network.h"
#include <iostream>
#include <map>


int main(int argc, char** arg){
    int inputs = 8;
    int hidden = 4;
    int memory = 2; //LSTM neurons
    int hidden_layers = 2;
    int outputs = 3;
    std::vector<int> connections; // I'll actually populate this at some point

    NetworkGenetic ConstructedNetwork(inputs, hidden, memory, outputs, hidden_layers,  connections);
    ConstructedNetwork.allocateHostAndGPUObjects(0.25, 0.85);

    int sampleRate, numberOfSites, SLEN;
    std::cin>>sampleRate>>numberOfSites>>SLEN;

    std::vector<double> sitesData;

    for (int i=0; i < SLEN; i){
        sitesData.push_back(0);
        std::cin>>sitesData.at(i);
    }
    int ret = ConstructedNetwork.init(sampleRate, numberOfSites, sitesData);
    std::cout<<ret<<std::endl;
    int doTraining;
    std::cin>>doTraining;
    if(!ConstructedNetwork.checkForWeights("/weights.bin"))
            ConstructedNetwork.initializeWeights();
    if (doTraining == 1)
    {
        int gtf_site, gtf_hour;
        double gtf_lat, gtf_long, gtf_mag, gtf_dist;
        std::cin>>gtf_site>>gtf_hour>>gtf_lat>>gtf_long>>gtf_mag>>gtf_dist;
        ConstructedNetwork.doingTraining(gtf_site, gtf_hour, gtf_lat, gtf_long, gtf_mag, gtf_dist);
    }
    while(1)
    {
        int DLEN, QLEN;
        int hour;
        double k;
        std::vector<int> data;
        std::vector<double> globalQuakes(5);
        std::cin>>hour;
        if(hour== -1)
            break;
        std::cin>>DLEN;
        for(int i=0; i<DLEN; i){
            data.push_back(0);
            std::cin>>data.at(i);
        }
       std::cin>>k>>QLEN;
       std::vector<double> tmpQuakes;
        for(int i=0; i<QLEN; i){
            tmpQuakes.push_back(0);
            std::cin>>tmpQuakes.at(i);
        }
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

            double * ret = ConstructedNetwork.forecast(hour, &data, k, &globalQuakes);
            std::cout<<2160*numberOfSites;
            for(int i=0; i<=2160*numberOfSites; i++){
                std::cout<<ret[i]<<std::endl;
            }
            std::cout.flush();
        }
    if(doTraining == 1)
        ConstructedNetwork.storeWeights("/weights.bin");
}
