#include <kernelDefs.h>

__global__ void NetKern(kernelArray<double> weights, kernelArray<int> params, kernelArray<double> globalQuakes,
                        kernelArray<int> inputVal, kernelArray<double> siteData, kernelArray<double> answers,
                        kernelArray<std::pair<int, int> > connections, double Kp, int sampleRate,int numOfSites,
                        int* site_offset, int* channel_offset,int hour, kernelArray<double> meanCh, kernelArray<double> stdCh, size_t device_offset){
    extern __shared__ float scratch[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    typedef std::pair<int, int>*  connectPairMatrix;

    float *When = &scratch[numOfSites*threadIdx.x];
    float *HowCertain = &scratch[numOfSites*blockDim.x + numOfSites*threadIdx.x];
    float *CommunityMag = &scratch[numOfSites*blockDim.x*2 + numOfSites*threadIdx.x];
    for(int i=0; i<numOfSites; i++){
        When[i]=0;
        HowCertain[i]=0;
        CommunityMag[i]=1;
    }
    int ind = params.array[10]; // number of individuals on device
    int startOfInput = params.array[12] + idx + device_offset; // 6 is the offset to the start of the input neurons
    int startOfHidden = params.array[13] + idx + device_offset;
    int startOfMem = params.array[14] + idx+ device_offset;
    int startOfMemGateIn = params.array[15] + idx + device_offset;
    int startOfMemGateOut = params.array[16] + idx + device_offset;
    int startOfMemGateForget = params.array[17] + idx + device_offset;
    int startOfOutput = params.array[18] + idx + device_offset;
    int startOfFitness = params.array[19] + idx + device_offset;
    double *input = &weights.array[startOfInput];
    double *hidden = &weights.array[startOfHidden];
    double *mem = &weights.array[startOfMem];
    double *memGateIn = &weights.array[startOfMemGateIn];
    double *memGateOut = &weights.array[startOfMemGateOut];
    double *memGateForget = &weights.array[startOfMemGateForget];
    double *outputs = &weights.array[startOfOutput];
    double *fitness = &weights.array[startOfFitness];
    for(int i=0; i<50; i++){
        double CommunityLat = 0;
        double CommunityLon = 0;
        for(int j=0; j<sampleRate; j++){//sitesWeighted Lat/Lon values are determined based on all previous zsites mag output value.
            CommunityLat += siteData.array[j*2]*CommunityMag[j];
            CommunityLon += siteData.array[j*2+1]*CommunityMag[j];
        }
        CommunityLat = CommunityLat/numOfSites;
        CommunityLon = CommunityLon/numOfSites;
        for(int j=0; j<numOfSites; j++){ //each site is run independently of others, but shares an output from the previous step

            double latSite = siteData.array[j*2];
            double lonSite = siteData.array[j*2+1];
            double avgLatGQuake = globalQuakes.array[0];
            double avgLonGQuake = globalQuakes.array[0];
            double GQuakeAvgMag = globalQuakes.array[2];
            double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                        1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            //                int n =0; // n is the weight number
            for(int k=0; k<3; k++){
                input[k+ind] = normalize(inputVal.array[site_offset[j]+channel_offset[k]+i], meanCh.array[k], stdCh.array[k]);//channel 1
            }
            input[3+ind] = shift(GQuakeAvgdist, 40075.1, 0);
            input[4+ind] = shift(GQuakeAvgBearing, 360, 0);
            input[5+ind] = shift(GQuakeAvgMag, 9.5, 0);
            input[6+ind] = shift(Kp, 10, 0);
            input[7+ind] = shift(CommunityDist,40075.1/2, 0);
            input[8+ind] = shift(CommunityBearing, 360, 0);
            //lets reset all neuron values for this new timestep (except memory neurons)
            //            for(int gate=0; gate<params.array[11]; gate++){
            //                memGateIn[gate] = 0;
            //                memGateOut[gate] = 0;
            //                memGateForget[gate] = 0;
            //            }
            //            for(int hid=0; hid<params.array[10]; hid++){
            //                hidden[hid] = 0;
            //            }
            //            for(int out=0; out<params.array[12]; out++){
            //                outputs[out] = 0;
            //            }

            //            //now that everything that should be zeroed is zeroed, lets start the network.
            //            //mem gates & LSTM nodes --
            //            for(int gate = 0; gate<params.array[11]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateIn
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it); // this needs to be created to use the iterator it correctly.
            //                    if(itr.second == gate+startOfMemGateIn && itr.second < startOfHidden){ //for inputs
            //                        memGateIn[gate] += input[itr.first-startOfInput]*weights.array[ind + n++]; // memGateIn vect starts at 0
            //                    }
            //                    else if(itr.second == gate+startOfMemGateIn && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateIn[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for memGateOut
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == gate+startOfMemGateOut && itr.second < startOfHidden){//for inputs
            //                        memGateOut[gate] += input[itr.first-startOfInput]*weights.array[ind + n++];
            //                    }
            //                    else if(itr.second == gate+startOfMemGateOut && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateOut[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//for  memGateForget
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == gate+startOfMemGateForget && itr.second < startOfHidden){//for inputs
            //                        memGateForget[gate] += input[itr.first - startOfInput]*weights.array[ind + n++];
            //                    }
            //                    else if(itr.second == gate+startOfMemGateForget && itr.second >startOfHidden && itr.second <startOfMem){//for hidden neurons
            //                        memGateForget[gate] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                memGateIn[gate] = ActFunc(memGateIn[gate]);
            //                memGateOut[gate] = ActFunc(memGateOut[gate]);
            //                memGateForget[gate] = ActFunc(memGateForget[gate]);
            //            }
            //            //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
            //            for (int gate = 0; gate<params.array[11]; gate++){ // if memGateIn is greater than 0.3, then let mem = the sum inputs attached to memGateIn
            //                if(memGateIn[gate] > 0.5){ //gate -startOfMemGateIn = [0, num of mem neurons]
            //                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                        if(itr.second == gate+startOfMemGateIn && itr.first < gate+startOfHidden){//only pass inputs
            //                            mem[gate] += input[itr.first-startOfInput]; // no weights attached, but the old value stored here is not removed.
            //                        }
            //                    }
            //                }
            //                if(memGateForget[gate] > 0.5){// if memGateForget is greater than 0.5, then tell mem to forget
            //                    mem[gate] = 0;
            //                }
            //                //if memGateForget fires, then memGateOut will output nothing.
            //                if(memGateOut[gate] > 0.5){//if memGateOut is greater than 0.3, let the nodes mem is connected to recieve mem
            //                    for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                        std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                        if(itr.first == gate+startOfMem){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
            //                            hidden[itr.second-startOfHidden] += mem[gate];
            //                        }
            //                    }
            //                }
            //            }

            //            // hidden neuron nodes --
            //            for(int hid=0; hid<params.array[10]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){ // Add the inputs to the hidden neurons
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == hid+startOfHidden && itr.first < startOfHidden && itr.first >= startOfInput){ // if an input connects with this hidden neuron
            //                        hidden[hid] += input[itr.first]*weights.array[ind + n++];
            //                    }
            //                }
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){//add other hidden neuron inputs to each hidden neuron (if applicable)
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == hid+startOfHidden && itr.first < startOfMem && itr.first >= startOfHidden){
            //                        hidden[hid] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                hidden[hid] += 1*weights.array[ind + n++]; // add bias
            //                hidden[hid] = ActFunc(hidden[hid]); // then squash itr.
            //            }
            //            //output nodes --

            //            for(int out =0; out<params.array[12]; out++){// add hidden neurons to the output nodes
            //                for(connectPairMatrix it = connections.array; it!= connections.array+connections.size; ++it){
            //                    std::pair<int, int>itr = static_cast<std::pair<int, int> >(*it);
            //                    if(itr.second == out+startOfOutput){
            //                        outputs[out] += hidden[itr.first-startOfHidden]*weights.array[ind + n++];
            //                    }
            //                }
            //                outputs[out] += 1*weights.array[ind + n++]; // add bias
            //                outputs[out] = ActFunc(outputs[out]);// then squash itr.
            //            }

            //            When[j*threadIdx.x] += outputs[0]*((2160-hour)-hour)+2160-hour; // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            //            HowCertain[j*threadIdx.x] += outputs[1];
            //            CommunityMag[j*threadIdx.x] =  outputs[2]; // set the next sets communityMag = output #3.
        }
    }
    //    for(int j=0; j<numOfSites; j++){ // now lets get the average when and howcertain values.
    //        When[j*threadIdx.x] = When[j*threadIdx.x]/3600*sampleRate;
    //        HowCertain[j*threadIdx.x] = HowCertain[j*threadIdx.x]/3600*sampleRate;
    //    }
    /*calculate performance for this individual - score = 1/(abs(whenGuess-whenReal)*distToQuake), for whenGuess = when[j] where HowCertain is max for set.
    distToQuake is from the current sites parameters, it emphasizes higher scores for the closest site, a smaller distance is a higher score. */
    //    int maxCertainty=0;
    //    double whenGuess=0;
    //    double latSite=0;
    //    double lonSite=0;
    //    for(int j=0; j<numOfSites; j++){
    //        if(HowCertain[j*threadIdx.x] > maxCertainty){
    //            whenGuess = When[j*threadIdx.x];
    //            latSite = siteData.array[j*2];
    //            lonSite = siteData.array[j*2+1];
    //        }
    //    }
    //    double SiteToQuakeDist = distCalc(latSite, lonSite, answers.array[1], answers.array[2]); // [2] is latitude, [3] is longitude.
    //    double fitness = 1/(abs(whenGuess - answers.array[0]-hour)*SiteToQuakeDist);//larger is better, negative numbers are impossible.
    //    weights.array[ind + params.array[2]-1] = fitness; // set the fitness number for the individual.
}
