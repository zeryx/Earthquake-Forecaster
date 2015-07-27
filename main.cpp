#include <vector>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <dlib/bayes_utils.h>


#include <iostream>
#include "test.h"

int main()
{
    std::cout << run() << std::endl;
    return 0;
}

//int quakeInit(int sampleRate, int Sites, std::vector<double>sitesData){
//    return 1;
//}

//int main(int argc, char* argv[])
//{
//    int sampleRate, S, SLEN;
//    std::cin>>sampleRate>>S>>SLEN;

//    std::vector<double> sitesData;

//    for (int i=0; i < SLEN; i++){
//        sitesData.push_back(0);
//        std::cin>>sitesData.at(i);
//    }
//    int ret = quakeInit(sampleRate, S, sitesData);
//    std::cout<<ret<<std::endl;
//    int doTraining;
//    std::cin>>doTraining;
//    if (doTraining == 1)
//    {
//        int gtf_site, gtf_hour;
//        double gtf_lat, gtf_long, gtf_mag, gtf_dist;
//        std::cin>>gtf_site>>gtf_hour>>gtf_lat>>gtf_long>>gtf_mag>>gtf_dist;
//    }
//    while(1)
//    {
//        int DLEN, QLEN;
//        int hour;
//        double k;
//        std::vector<int> data;
//        std::vector<double> globalQuakes;
//        std::cin>>hour;
//        if(hour== -1)
//            break;
//        std::cin>>DLEN;
//        for(int i=0; i<DLEN; i++){
//            data.push_back(0);
//            std::cin>>data.at(i);
//        }
//       std::cin>>k>>QLEN;
//        for(int i=0; i<QLEN; i++){
//            globalQuakes.push_back(0);
//            std::cin>>globalQuakes.at(i);
//        }
//        int arraylength = 2160*S;
//        std::cout<<arraylength<<std::endl;
//        srand((long)time(NULL));
//        for(int i=0; i<2160*S; i++){

//            std::cout<<(double)1/(double)rand()<<std::endl;
//        }
//        std::cout.flush();
//    }
//    return 0;
//}

