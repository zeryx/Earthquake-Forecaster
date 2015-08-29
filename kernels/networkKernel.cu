#include <kernelDefs.h>
//using
extern __constant__ int inputData[];
extern __constant__ double answers[];
extern __constant__ double globalQuakes[];
extern __constant__ double siteData[];
extern __constant__ double Kp;
extern __constant__ int site_offset[];
extern __constant__ int channel_offset[];
extern __constant__ int trainingsize;
//endof using

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params, devicePair<dcon, dcon>* connections, int hour, kernelArray<double> meanCh,
                        kernelArray<double> stdCh, size_t device_offset){
    const int tix = threadIdx.x;
    extern __shared__ devicePair<dcon, dcon> shdConnect[];
    for(int i=0; i<params.array[26]; i=i+blockDim.x){
        if(tix+i<params.array[26]){
            shdConnect[tix+i].first.first = connections[tix+i].first.first;
            shdConnect[tix+i].first.second = connections[tix+i].first.second;
            shdConnect[tix+i].second.first = connections[tix+i].second.first;
            shdConnect[tix+i].second.second = connections[tix+i].second.second;
        }
    }
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int ind = params.array[10]; // number of individuals on device
    const int weightsOffset = params.array[11] + idx + device_offset;
    const int inputOffset = params.array[12] + idx + device_offset; // 6 is the offset to the start of the input neurons
    const int hiddenOffset = params.array[13] + idx + device_offset;
    const int memOffset = params.array[14] + idx + device_offset;
    const int memGateInOffset = params.array[15] + idx + device_offset;
    const int memGateOutOffset = params.array[16] + idx + device_offset;
    const int memGateForgetOffset = params.array[17] + idx + device_offset;
    const int outputOffset = params.array[18] + idx + device_offset;
    const int fitnessOffset = params.array[19] + idx + device_offset;
    const int communityMagOffset = params.array[20] +idx +device_offset;
    const int whenOffset = params.array[21] + idx + device_offset;
    const int howCertainOffset = params.array[22] + idx + device_offset;
    const int ageOffset = params.array[25] + idx + device_offset;
    Vec.array[ageOffset] += 1; //this indvidiual has existed for 1 more iteration.

    //reset values from previous individual.
    //community magnitude is not set, as this needs to be continued.
    for(int i=0; i<params.array[23]; i++){
        Vec.array[whenOffset +i*ind] = 0;
        Vec.array[howCertainOffset +i*ind] =0;
    }

    for(int i=0; i<trainingsize; i++){
        float CommunityLat = 0;
        float CommunityLon = 0;
        for(int j=0; j<params.array[23]; j++){//sitesWeighted Lat/Lon values are determined based on all previous zsites mag output value.
            CommunityLat += siteData[j*2]*Vec.array[communityMagOffset+j*ind];
            CommunityLon += siteData[j*2+1]*Vec.array[communityMagOffset+j*ind];
        }
        CommunityLat = CommunityLat/params.array[23];
        CommunityLon = CommunityLon/params.array[23];
        for(int j=0; j<params.array[23]; j++){ //each site is run independently of others, but shares an output from the previous step

            float latSite = siteData[j*2];
            float lonSite = siteData[j*2+1];
            float avgLatGQuake = globalQuakes[0];
            float avgLonGQuake = globalQuakes[1];
            float GQuakeAvgMag = globalQuakes[3];
            float GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            float GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            float CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            float CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);
            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                        1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0; // n is the weight number
            for(int k=0; k<3; k++){
                Vec.array[inputOffset+k*ind] = normalize(inputData[site_offset[j]+channel_offset[k]+i], meanCh.array[k], stdCh.array[k]);//channel 1
            }
            Vec.array[inputOffset+3*ind] = shift(GQuakeAvgdist, 40075.1, 0);
            Vec.array[inputOffset+4*ind] = shift(GQuakeAvgBearing, 360, 0);
            Vec.array[inputOffset+5*ind] = shift(GQuakeAvgMag, 9.5, 0);
            Vec.array[inputOffset+6*ind] = shift(Kp, 10, 0);
            Vec.array[inputOffset+7*ind] = shift(CommunityDist,40075.1, 0);
            Vec.array[inputOffset+8*ind] = shift(CommunityBearing, 360, 0);
            //lets reset all neuron values for this new timestep (except memory neurons)
            for(int gate=0; gate<params.array[5]; gate++){
                Vec.array[memGateInOffset+gate*ind] = 0;
                Vec.array[memGateOutOffset+gate*ind] = 0;
                Vec.array[memGateForgetOffset+gate*ind] = 0;
            }
            for(int hid=0; hid<params.array[4]; hid++){
                Vec.array[hiddenOffset+hid*ind] = 0;
            }
            for(int out=0; out<params.array[9]; out++){
                Vec.array[outputOffset+out*ind] = 0;
            }

            //now that everything that should be zeroed is zeroed, lets start the network.
            //mem gates & LSTM nodes --
            for(int gate = 0; gate<params.array[6]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                for(int pair=0; pair<params.array[26]; pair++){
                    //for memGateIn
                    if(shdConnect[pair].second.first == typeMemGateIn && shdConnect[pair].second.second == gate && shdConnect[pair].first.first ==typeHidden){ //for inputs
                        Vec.array[memGateInOffset+gate*ind] += Vec.array[inputOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind]; // memGateIn vect starts at 0
                    }
                    else if(shdConnect[pair].second.first == typeMemGateIn && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeHidden){//for hidden neurons
                        Vec.array[memGateInOffset+gate*ind] += Vec.array[hiddenOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind];
                    }
                }
                Vec.array[memGateInOffset+gate*ind] = ActFunc(Vec.array[memGateInOffset+gate*ind]);
            }
            //for memGateOut

            for(int gate = 0; gate<params.array[7]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                for(int pair=0; pair<params.array[26]; pair++){
                    if(shdConnect[pair].second.first == typeMemGateOut && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeInput){//for inputs
                        Vec.array[memGateOutOffset+gate*ind] += Vec.array[inputOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind];
                    }
                    else if(shdConnect[pair].second.first == typeMemGateOut && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeHidden){//for hidden neurons
                        Vec.array[memGateOutOffset+gate*ind] += Vec.array[hiddenOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind];
                    }
                }
                Vec.array[memGateOutOffset+gate*ind] = ActFunc(Vec.array[memGateOutOffset+gate*ind]);

            }
            //for  memGateForget
            for(int gate = 0; gate<params.array[8]; gate++){//calculate memory gate node values, you can connect inputs & hidden neurons to them.
                for(int pair=0; pair<params.array[26]; pair++){
                    if(shdConnect[pair].second.first == typeMemGateForget && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeInput){//for inputs
                        Vec.array[memGateForgetOffset+gate*ind] += Vec.array[inputOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind];
                    }
                    else if(shdConnect[pair].second.first == typeMemGateForget && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeHidden){//for hidden neurons
                        Vec.array[memGateForgetOffset+gate*ind] += Vec.array[hiddenOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset+(n++)*ind];
                    }
                }
                Vec.array[memGateForgetOffset+gate*ind] = ActFunc(Vec.array[memGateForgetOffset+gate*ind]);
            }
            //since we calculated the values for memGateIn and memGateOut, and MemGateForget..
            for (int gate = 0; gate<params.array[6]; gate++){ // if memGateIn is greater than 0.5, then let mem = the sum inputs attached to memGateIn
                if(Vec.array[memGateInOffset+gate*ind] > 0.5){ //gate -memGateInOffset = [0, num of mem neurons]
                    for(int pair=0; pair<params.array[26]; pair++){
                        if(shdConnect[pair].second.first == typeMemGateIn && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeInput){//only pass inputs
                            Vec.array[memOffset+gate*ind] += Vec.array[inputOffset+(shdConnect[pair].first.second)*ind]; // no Vec attached, but the old value stored here is not removed.
                        }
                    }
                }
            }
            for (int gate = 0; gate<params.array[7]; gate++){ // if memGateForget is greater than 0.5, then tell mem to forget
                if(Vec.array[memGateForgetOffset+gate*ind] > 0.5){
                    for(int pair=0; pair<params.array[26]; pair++){
                        if(shdConnect[pair].second.first == typeMemGateForget && shdConnect[pair].second.second == gate && shdConnect[pair].first.first == typeMemory){
                            Vec.array[memOffset+shdConnect[pair].first.second*ind] =0;
                        }
                    }
                }
            }
            //if memGateForget fires, then memGateOut will output nothing.
            for (int gate = 0; gate<params.array[7]; gate++){//if memGateOut is greater than 0.5, let the nodes mem is connected to recieve mem
                if(Vec.array[memGateOutOffset+gate*ind] > 0.5){
                    for(int pair=0; pair<params.array[26]; pair++){
                        if(shdConnect[pair].first.first == typeMemory && shdConnect[pair].first.second == gate && shdConnect[pair].second.first == typeHidden){// since mem node: memIn node : memOut node = 1:1:1, we can do this.
                            Vec.array[hiddenOffset+(shdConnect[pair].second.second)*ind] += Vec.array[memOffset+gate*ind];
                        }
                    }
                }
            }

            // hidden neuron nodes --
            for(int hid=0; hid<params.array[4]; hid++){ // for all hidden neurons at layer 1, lets sum the inputs, the memory values were already added.
                for(int pair=0; pair<params.array[26]; pair++){ // Add the inputs to the hidden neurons
                    if(shdConnect[pair].second.first == typeHidden && shdConnect[pair].second.first == hid && shdConnect[pair].first.first == typeInput){ // if an input connects with this hidden neuron
                        Vec.array[hiddenOffset+hid*ind] += Vec.array[inputOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset + (n++)*ind];
                    }
                    else if(shdConnect[pair].second.first == typeHidden && shdConnect[pair].second.second == hid && shdConnect[pair].first.first == typeHidden){
                        Vec.array[hiddenOffset+hid*ind] += Vec.array[hiddenOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset + (n++)*ind];
                    }
                }
                Vec.array[hiddenOffset+hid*ind] += 1*Vec.array[weightsOffset + (n++)*ind]; // add bias
                Vec.array[hiddenOffset+hid*ind] = ActFunc(Vec.array[hiddenOffset+hid*ind]); // then squash it.
            }
            //output nodes --
            for(int out =0; out<params.array[9]; out++){// add hidden neurons to the output nodes
                for(int pair=0; pair<params.array[26]; pair++){
                    if(shdConnect[pair].second.first == typeOutput && shdConnect[pair].second.second == out && shdConnect[pair].first.first == typeHidden){
                        Vec.array[outputOffset+out*ind] += Vec.array[hiddenOffset+(shdConnect[pair].first.second)*ind]*Vec.array[weightsOffset + (n++)*ind];
                    }
                }
                Vec.array[outputOffset+out*ind] += 1*Vec.array[weightsOffset + (n++)*ind]; // add bias
                Vec.array[outputOffset+out*ind] = ActFunc(Vec.array[outputOffset+out*ind]);// then squash it.
            }

            Vec.array[whenOffset+j*ind] += Vec.array[outputOffset+0*ind]*((2160-hour)-hour)+2160-hour; // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            Vec.array[howCertainOffset+j*ind] += Vec.array[outputOffset+1*ind];
            Vec.array[communityMagOffset+j*ind] =  Vec.array[outputOffset+2*ind]; // set the next sets communityMag = output #3.
        }
    }
    for(int j=0; j<params.array[23]; j++){ // now lets get the average when and howcertain values.
        Vec.array[whenOffset+j*ind] = Vec.array[whenOffset+j*ind]/trainingsize;
        Vec.array[howCertainOffset+j*ind] = Vec.array[howCertainOffset+j*ind]/trainingsize;
    }
    /*calculate score for this individual during this round, current scoring mechanism is - e^(-(abs(whenGuess-whenAns)+distToCorrectSite)), closer to 1 the better.   */
    float maxCertainty=0;
    float whenGuess=0;
    float guessLat=0;
    float guessLon=0;
    for(int j=0; j<params.array[23]; j++){
        if(Vec.array[howCertainOffset+j*ind] > maxCertainty){
            maxCertainty = Vec.array[howCertainOffset+j*ind];
            whenGuess = Vec.array[whenOffset+j*ind];
            guessLat = siteData[j*2];
            guessLon = siteData[j*2+1];
        }
    }
    float ansLat = siteData[(int)answers[0]*2];
    float ansLon = siteData[(int)answers[0]*2+1];
    int whenAns = (int)answers[1] - hour;
    double oldFit = Vec.array[fitnessOffset];
    Vec.array[fitnessOffset] = scoreFunc(whenGuess, whenAns, guessLat, guessLon, ansLat, ansLon, oldFit); //we take the average beacuse consistency is more important than being really good at this particular hour.
}
