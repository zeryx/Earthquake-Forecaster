#include <prep.h>
#include <getsys.h>
#include <connections.h>
#include <iostream>
#include <vector>
#include <connections.h>
#include <map>
#include <cstdlib>

int main(int argc, char** arg){
    prep start;
    if(!start.readOrders("orders.json")){
        std::cerr<<"couldn't read orders.json file..."<<std::endl;
        return false;
    }
    if(!start.readNetParmeters("net.json")){
        std::cerr<<"couldn't read net.json file..."<<std::endl;
        return false;
    }
    int sampleRate, numberOfSites, SLEN;
    std::cin>>sampleRate>>numberOfSites>>SLEN;
    std::vector<double> sitesData;

    for (int i=0; i < SLEN; i++){
        sitesData.push_back(0);
        std::cin>>sitesData.at(i);
    }
    bool initRet= start.init(sampleRate, numberOfSites, &sitesData);
    std::cout<<initRet<<std::endl;
    int doTraining;
    std::cin>>doTraining;
    if (doTraining == 1)
    {
        int gtf_site, gtf_hour;
        double gtf_lat, gtf_long, gtf_mag, gtf_dist;
        std::cin>>gtf_site>>gtf_hour>>gtf_lat>>gtf_long>>gtf_mag>>gtf_dist;
        start.doingTraining(gtf_site, gtf_hour, gtf_lat, gtf_long, gtf_mag, gtf_dist);

        if(start.checkForGenomes("/home/ubuntu/mount/genome/weights.bin")){

            std::cerr<<"running a hot start"<<std::endl;

            start.hotStart("/home/ubuntu/mount/genome/weights.bin");
        }
        else{
            std::cerr<<"running a cold start"<<std::endl;

            start.coldStart();
        }
    }
    while(1)
    {
        int DLEN, QLEN;
        int hour;
        double Kp;
        std::vector<int> data;
        std::vector<double> globalQuakes(5, 0);
        std::vector<double> tmpQuakes;
        std::vector<double> retM(2160*numberOfSites, 0);


        std::cin>>hour;
        if(hour== -1 || hour == 500)
            break;
        std::cin>>DLEN;
        for(int i=0; i<DLEN; i++){
            data.push_back(0);
            std::cin>>data.at(i);
            if(data.at(i) == -1){
                if(i>0)
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
        start.forecast(retM, hour, data, Kp, globalQuakes);
        std::cout<<retM.size()<<std::endl;
        for(unsigned int i=0; i<retM.size(); i++){
            std::cout<<retM.at(i)<<std::endl;
        }
        std::cout.flush();
    }
    if(doTraining == 1){
        start.EndOfTrial("/home/ubuntu/mount/genome/weights.bin");
    }
    return 1;
}
