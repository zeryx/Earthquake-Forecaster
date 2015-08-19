#include <kernelDefs.h>
__constant__ int input[40*20*3];
__constant__ int site_offset[20];
__constant__ int channel_offset[3];

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params, kernelArray<double> globalQuakes,
                        kernelArray<double> siteData, kernelArray<double> answers,
                        kernelArray<std::pair<int, int> > connections, double Kp,int numOfSites,
                        int hour, kernelArray<double> meanCh, kernelArray<double> stdCh, size_t device_offset){

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    typedef std::pair<int, int>*  connectPairMatrix;
    int ind = params.array[10]; // number of individuals on device
    int startOfInput = params.array[12] + idx + device_offset; // 6 is the offset to the start of the input neurons
    int startOfHidden = params.array[13] + idx + device_offset;
    int startOfMem = params.array[14] + idx+ device_offset;
    int startOfMemGateIn = params.array[15] + idx + device_offset;
    int startOfMemGateOut = params.array[16] + idx + device_offset;
    int startOfMemGateForget = params.array[17] + idx + device_offset;
    int startOfOutput = params.array[18] + idx + device_offset;
    int startOfFitness = params.array[19] + idx + device_offset;
    int startOfCommunityMag = params.array[20] +idx +device_offset;
    int startOfWhen = params.array[21] + idx + device_offset;
    int startOfHowCertain = params.array[22] + idx + device_offset;
    for(int i=0; i<numOfSites; i++){
        Vec.array[startOfWhen+i*ind]=0;
        Vec.array[startOfHowCertain+i*ind]=0;
        Vec.array[startOfCommunityMag+i*ind]=1;
    }
    for(int i=0; i<40; i++){
        double CommunityLat = 0;
        double CommunityLon = 0;
        for(int j=0; j<numOfSites; j++){//sitesWeighted Lat/Lon values are determined based on all previous zsites mag output value.
            CommunityLat += siteData.array[j*2]*Vec.array[startOfCommunityMag+j*ind];
            CommunityLon += siteData.array[j*2+1]*Vec.array[startOfCommunityMag+j*ind];
        }
        CommunityLat = CommunityLat/numOfSites;
        CommunityLon = CommunityLon/numOfSites;
        for(int j=0; j<numOfSites; j++){ //each site is run independently of others, but shares an output from the previous step

            double latSite = siteData.array[j*2];
            double lonSite = siteData.array[j*2+1];
            double avgLatGQuake = globalQuakes.array[0];
            double avgLonGQuake = globalQuakes.array[1];
            double GQuakeAvgMag = globalQuakes.array[3];
            double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                        1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0; // n is the weight number
            for(int k=0; k<3; k++){
                Vec.array[startOfInput+k*ind] = normalize(input[site_offset[j]+channel_offset[k]+i], meanCh.array[k], stdCh.array[k]);//channel 1
            }
            Vec.array[startOfInput+3*ind] = shift(GQuakeAvgdist, 40075.1, 0);
            Vec.array[startOfInput+4*ind] = shift(GQuakeAvgBearing, 360, 0);
            Vec.array[startOfInput+5*ind] = shift(GQuakeAvgMag, 9.5, 0);
            Vec.array[startOfInput+6*ind] = shift(Kp, 10, 0);
            Vec.array[startOfInput+7*ind] = shift(CommunityDist,40075.1, 0);
            Vec.array[startOfInput+8*ind] = shift(CommunityBearing, 360, 0);
            //lets reset all neuron values for this new timestep (except memory neurons)
            for(int gate=0; gate<params.array[5]; gate++){
                Vec.array[startOfMemGateIn+gate*ind] = 0;
                Vec.array[startOfMemGateOut+gate*ind] = 0;
                Vec.array[startOfMemGateForget+gate*ind] = 0;
            }
            for(int hid=0; hid<params.array[4]; hid++){
                Vec.array[startOfHidden+hid*ind] = 0;
            }
            for(int out=0; out<params.array[9]; out++){
                Vec.array[startOfOutput+out*ind] = 0;
            }

            //now that everything that should be zeroed is zeroed, lets start the network.
            //mem gates & LSTM nodes --
            for(int gate = 0; gate<params.array[5]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateIn
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it); // this needs to be created to use the iterator it correctly.
                    if(itr.second == gate+startOfMemGateIn && itr.second < startOfHidden){ //for inputs
                        Vec.array[startOfMemGateIn+gate*ind] += Vec.array[startOfInput+(itr.first-startOfInput)*ind]*Vec.array[(n++)*ind]; // memGateIn vect starts at 0
                    }
                    else if(itr.second == gate+startOfMemGateIn && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        Vec.array[startOfMemGateIn+gate*ind] += Vec.array[startOfHidden+(itr.first-startOfHidden)*ind]*Vec.array[(n++)*ind];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateOut
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                    if(itr.second == gate+startOfMemGateOut && itr.second < startOfHidden){//for inputs
                        Vec.array[startOfMemGateOut+gate*ind] += Vec.array[startOfInput+(itr.first-startOfInput)*ind]*Vec.array[(n++)*ind];
                    }
                    else if(itr.second == gate+startOfMemGateOut && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        Vec.array[startOfMemGateOut+gate*ind] += Vec.array[startOfHidden+(itr.first-startOfHidden)*ind]*Vec.array[(n++)*ind];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for  memGateForget
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                    if(itr.second == gate+startOfMemGateForget && itr.second < startOfHidden){//for inputs
                        Vec.array[startOfMemGateForget+gate*ind] += Vec.array[startOfInput+itr.first - startOfInput]*Vec.array[(n++)*ind];
                    }
                    else if(itr.second == gate+startOfMemGateForget && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
                        Vec.array[startOfMemGateForget+gate*ind] += Vec.array[startOfHidden+(itr.first-startOfHidden)*ind]*Vec.array[(n++)*ind];
                    }
                }
                Vec.array[startOfMemGateIn+gate*ind] = ActFunc(Vec.array[startOfMemGateIn+gate*ind]);
                Vec.array[startOfMemGateOut+gate*ind] = ActFunc(Vec.array[startOfMemGateOut+gate*ind]);
                Vec.array[startOfMemGateForget+gate*ind] = ActFunc(Vec.array[startOfMemGateForget+gate*ind]);
            }
            //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
            for (int gate = 0; gate<params.array[5]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
                if(Vec.array[startOfMemGateIn+gate*ind] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                        if(itr.second == gate+startOfMemGateIn && itr.first < gate+startOfHidden){//only pass inputs
                            Vec.array[startOfMem+gate*ind] += Vec.array[startOfInput+(itr.first-startOfInput)*ind]; // no Vec attached, but the old value stored here is not removed.
                        }
                    }
                }
                if(Vec.array[startOfMemGateForget+gate*ind] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
                    Vec.array[startOfMem+gate*ind] = 0;
                }
                //if memGateForget fires, then memGateOut will output nothing.
                if(Vec.array[startOfMemGateOut+gate*ind] > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                        if(itr.first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
                            Vec.array[startOfHidden+itr.second] += Vec.array[startOfMem+gate*ind];
                        }
                    }
                }
            }

            // hidden neuron nodes --
            for(int hid=0; hid<params.array[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){ // Add the inputs to the hidden neurons
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                    if(itr.second == hid+startOfHidden && itr.first < startOfHidden && itr.first >= startOfInput){ // if an input connects with this hidden neuron
                        Vec.array[startOfHidden+hid*ind] += Vec.array[startOfInput+itr.first*ind]*Vec.array[(n++)*ind];
                    }
                }
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                    if(itr.second == hid+startOfHidden && itr.first < startOfMem && itr.first >= startOfHidden){
                        Vec.array[startOfHidden+hid*ind] += Vec.array[startOfHidden+(itr.first-startOfHidden)*ind]*Vec.array[(n++)*ind];
                    }
                }
                Vec.array[startOfHidden+hid*ind] += 1*Vec.array[(n++)*ind]; // add bias
                Vec.array[startOfHidden+hid*ind] = ActFunc(Vec.array[startOfHidden+hid*ind]); // then squash itr.
            }
            //output nodes --

            for(int out =0; out<params.array[9]; out++){// add hidden neurons to the output nodes
                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
                    if(itr.second == out+startOfOutput){
                        Vec.array[startOfOutput+out*ind] += Vec.array[startOfHidden+(itr.first-startOfHidden)*ind]*Vec.array[(n++)*ind];
                    }
                }
                Vec.array[startOfOutput+out*ind] += 1*Vec.array[(n++)*ind]; // add bias
                Vec.array[startOfOutput+out*ind] = ActFunc(Vec.array[startOfOutput+out*ind]);// then squash itr.
            }

            Vec.array[startOfWhen+j*ind] += Vec.array[startOfOutput+0*ind]*((2160-hour)-hour)+2160-hour; // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            Vec.array[startOfHowCertain+j*ind] += Vec.array[startOfOutput+1*ind];
            Vec.array[startOfCommunityMag+j*ind] =  Vec.array[startOfOutput+2*ind]; // set the next sets communityMag = output #3.
        }
    }
    for(int j=0; j<numOfSites; j++){ // now lets get the average when and howcertain values.
        Vec.array[startOfWhen+j*ind] = Vec.array[startOfWhen+j*ind]/40;
        Vec.array[startOfHowCertain+j*ind] = Vec.array[startOfHowCertain+j*ind]/40;
    }
    /*calculate performance for this individual - score = 1/(abs(whenGuess-whenReal)*distToQuake), for whenGuess = Vec.array[startOfWhen+j] where HowCertain is max for set.
    distToQuake is from the current sites parameters, it emphasizes higher scores for the closest site, a smaller distance is a higher score. */
    int maxCertainty=0;
    double whenGuess=0;
    double latSite=0;
    double lonSite=0;
    for(int j=0; j<numOfSites; j++){
        if(Vec.array[startOfHowCertain+j*ind] > maxCertainty){
            whenGuess = Vec.array[startOfWhen+j*ind];
            latSite = siteData.array[j*2];
            lonSite = siteData.array[j*2+1];
        }
    }
    double SiteToQuakeDist = distCalc(latSite, lonSite, answers.array[1], answers.array[2]); // [2] is latitude, [3] is longitude.
    Vec.array[startOfFitness] = 1/(fabs(whenGuess - answers.array[0]-hour)*SiteToQuakeDist);//larger is better, negative numbers are impossible.
}
