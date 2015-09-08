#include <kernelDefs.h>
#include <neuroFunc.h>
#include <utilFunc.h>
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

__global__ void NetKern(kernelArray<double> Vec, kernelArray<int> params, Order* commandQueue, int hour, kernelArray<double> meanCh,
                        kernelArray<double> stdCh, size_t device_offset){
    extern __shared__  Order sharedQueue[];
    const int tix = threadIdx.x;
    if(tix < params.array[26]){
        sharedQueue[tix].first = commandQueue[tix].first;
        sharedQueue[tix].second = commandQueue[tix].second;
    }
    __syncthreads();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    const int ind = params.array[10];

    const int weightsOffset = params.array[11] + idx + device_offset;
    const int inputOffset = params.array[12] + idx + device_offset;
    const int hiddenOffset = params.array[13] + idx + device_offset;
    const int memOffset = params.array[14] + idx + device_offset;
    const int memGateInOffset = params.array[15] + idx + device_offset;
    const int memGateOutOffset = params.array[16] + idx + device_offset;
    const int memGateForgetOffset = params.array[17] + idx + device_offset;
    const int outputOffset = params.array[18] + idx + device_offset;
    const int fitnessOffset = params.array[19] + idx + device_offset;
    const int communityMagOffset = params.array[20] +idx +device_offset;
    const int whenMinOffset = params.array[21] + idx + device_offset;
    const int howCertainOffset = params.array[22] + idx + device_offset;
    const int whenMaxOffset = params.array[25] + idx + device_offset;

    const double avgLatGQuake = globalQuakes[0];
    const double avgLonGQuake = globalQuakes[1];
    const double GQuakeAvgMag = globalQuakes[3];


    const double ansLat = siteData[(int)answers[0]*2];
    const double ansLon = siteData[(int)answers[0]*2+1];
    const int whenAns = (int)answers[1] - hour;

    //reset values from previous individual.
    //community magnitude is not set, as this needs to be continued.
    for(int i=0; i<params.array[23]; i++){
        Vec.array[whenMinOffset +i*ind] = 0;
        Vec.array[whenMaxOffset +i*ind] =0;
        Vec.array[howCertainOffset +i*ind] =0;
    }
    for(int i=0; i<trainingsize; i++){ // training size is a constant parameter for the size of each timestep

        double CommunityLat = 0;
        double CommunityLon = 0;

        for(int j=0; j<params.array[23]; j++){//sitesWeighted Lat/Lon values are determined based on all previous zsites mag output value.
            CommunityLat += siteData[j*2]*Vec.array[communityMagOffset+j*ind];
            CommunityLon += siteData[j*2+1]*Vec.array[communityMagOffset+j*ind];
        }

        CommunityLat = CommunityLat/params.array[23];
        CommunityLon = CommunityLon/params.array[23];


        for(int j=0; j<params.array[23]; j++){ //each site is run independently of others, but shares an output from the previous step

            const double latSite = siteData[j*2];
            const double lonSite = siteData[j*2+1];
            const double GQuakeAvgdist = distCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            const double GQuakeAvgBearing = bearingCalc(latSite, lonSite, avgLatGQuake, avgLonGQuake);
            const double CommunityDist = distCalc(latSite, lonSite, CommunityLat, CommunityLon);
            const double CommunityBearing = bearingCalc(latSite, lonSite, CommunityLat, CommunityLon);


            /* 3 outputs, 1 with an hour in the future when the earthquake will hit,
                        1 with the porbability of that earthquake happening (between [0,1]) and 1 with the sites magnitude (for community feedback) */
            int n =0; // n is the weight number

            for(int k=0; k<3; k++){
                Vec.array[inputOffset+k*ind] = normalize(inputData[site_offset[j]+channel_offset[k]+i], meanCh.array[k], stdCh.array[k]);//channels 1-3
            }

            Vec.array[inputOffset+3*ind] = shift(GQuakeAvgdist, 40075.1, 0);
            Vec.array[inputOffset+4*ind] = shift(GQuakeAvgBearing, 360, 0);
            Vec.array[inputOffset+5*ind] = shift(GQuakeAvgMag, 9.5, 0);
            Vec.array[inputOffset+6*ind] = shift(Kp, 10, 0);
            Vec.array[inputOffset+7*ind] = shift(CommunityDist,40075.1, 0);
            Vec.array[inputOffset+8*ind] = shift(CommunityBearing, 360, 0);
            //run the neuroCommand order tree
            for(int itr=0; itr< params.array[26]; itr++){//every order is sequential and run after the previous order to massively simplify the workload in this kernel.
                double tmp;
                //set stuff to zero
                if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[hiddenOffset+sharedQueue[itr].first.id*ind]);

                }

                else if(sharedQueue[itr].first.def == typeMemGateIn && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[memGateInOffset+sharedQueue[itr].first.id*ind]);

                }

                else if(sharedQueue[itr].first.def == typeMemGateOut && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[memGateOutOffset+sharedQueue[itr].first.id*ind]);

                }

                else if(sharedQueue[itr].first.def == typeMemGateForget && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[memGateForgetOffset+sharedQueue[itr].first.id*ind]);

                }

                else if(sharedQueue[itr].first.def == typeMemory && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[memOffset+sharedQueue[itr].first.id*ind]);

                }

                else if(sharedQueue[itr].first.def == typeOutput && sharedQueue[itr].second.def == typeZero){
                    neuroZero(Vec.array[outputOffset+sharedQueue[itr].first.id*ind]);

                }

                //first->second summations
                else if(sharedQueue[itr].first.def == typeInput && sharedQueue[itr].second.def == typeHidden){
                    tmp = Vec.array[inputOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[hiddenOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeInput && sharedQueue[itr].second.def == typeMemGateIn){
                    tmp = Vec.array[inputOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateInOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeInput && sharedQueue[itr].second.def == typeMemGateOut){
                    tmp = Vec.array[inputOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateOutOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeInput && sharedQueue[itr].second.def == typeMemGateForget){
                    tmp = Vec.array[inputOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateForgetOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeHidden){
                    tmp = Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[hiddenOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeMemGateIn){
                    tmp = Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateInOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeOutput){
                    tmp = Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[outputOffset + sharedQueue[itr].second.id*ind], tmp);

                }


                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeMemGateOut){
                    tmp = Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateOutOffset + sharedQueue[itr].second.id*ind], tmp);

                }

                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeMemGateForget){
                    Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateForgetOffset + sharedQueue[itr].second.id*ind], tmp);

                }


                //memory gates
                else if(sharedQueue[itr].first.def == typeInput && sharedQueue[itr].second.def == typeMemory && sharedQueue[itr].third.def == typeMemGateIn){

                    tmp = Vec.array[inputOffset+sharedQueue[itr].first.id*ind]; // squash inputs so as to not saturate hidden neurons
                    neuroSquash(tmp);

                    neuroMemGate(Vec.array[memGateInOffset+sharedQueue[itr].third.id*ind], tmp, Vec.array[memOffset+sharedQueue[itr].second.id*ind], 0.5);
                }

                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeMemory && sharedQueue[itr].third.def == typeMemGateIn){

                    tmp = Vec.array[hiddenOffset+sharedQueue[itr].first.id*ind];
                    neuroSquash(tmp);

                    neuroMemGate(Vec.array[memGateInOffset+sharedQueue[itr].third.id*ind], tmp, Vec.array[memOffset+sharedQueue[itr].second.id*ind], 0.5);
                }

                else if(sharedQueue[itr].first.def == typeOutput && sharedQueue[itr].second.def == typeMemory && sharedQueue[itr].third.def == typeMemGateIn){
                    neuroMemGate(Vec.array[memGateInOffset+sharedQueue[itr].third.id*ind],
                            Vec.array[outputOffset+sharedQueue[itr].first.id*ind],
                            Vec.array[memOffset + sharedQueue[itr].second.id*ind], 0.5);
                }

                else if(sharedQueue[itr].first.def == typeMemory && sharedQueue[itr].second.def == typeHidden && sharedQueue[itr].third.def == typeMemGateOut){
                    neuroMemGate(Vec.array[memGateOutOffset+sharedQueue[itr].third.id*ind],
                            Vec.array[memOffset+sharedQueue[itr].first.id*ind],
                            Vec.array[hiddenOffset+sharedQueue[itr].second.id*ind], 0.5);
                }

                else if(sharedQueue[itr].first.def == typeMemory && sharedQueue[itr].second.def == typeOutput && sharedQueue[itr].third.def == typeMemGateOut){
                    neuroMemGate(Vec.array[memGateOutOffset+sharedQueue[itr].third.id*ind],
                            Vec.array[memOffset+sharedQueue[itr].first.id*ind],
                            Vec.array[outputOffset+sharedQueue[itr].second.id*ind], 0.5);
                }

                else if(sharedQueue[itr].first.def == typeMemory && sharedQueue[itr].second.def == typeMemGateForget){
                    neuroMemForget(Vec.array[memGateForgetOffset+sharedQueue[itr].second.id*ind],
                            Vec.array[memOffset + sharedQueue[itr].first.id*ind], 0.5);
                }

                //bias
                else if(sharedQueue[itr].first.def == typeBias && sharedQueue[itr].second.def == typeHidden){
                    tmp = 1*Vec.array[weightsOffset + n++*ind];
                    neuroSum(Vec.array[hiddenOffset + sharedQueue[itr].second.id*ind], tmp);
                }

                else if(sharedQueue[itr].first.def == typeBias && sharedQueue[itr].second.def == typeMemGateIn){
                    tmp = 1*Vec.array[weightsOffset + n++*ind];
                    neuroSum(Vec.array[memGateInOffset + sharedQueue[itr].second.id*ind], tmp);
                }

                else if(sharedQueue[itr].first.def == typeBias && sharedQueue[itr].second.def == typeMemGateOut){
                    tmp = 1*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateOutOffset + sharedQueue[itr].second.id*ind], tmp);
                }

                else if(sharedQueue[itr].first.def == typeBias && sharedQueue[itr].second.def == typeMemGateForget){
                    tmp = 1*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[memGateForgetOffset + sharedQueue[itr].second.id*ind], tmp);
                }

                else if(sharedQueue[itr].first.def == typeBias && sharedQueue[itr].second.def == typeOutput){
                    tmp = 1*Vec.array[weightsOffset+n++*ind];
                    neuroSum(Vec.array[outputOffset + sharedQueue[itr].second.id*ind], tmp);
                }

                //squashing
                else if(sharedQueue[itr].first.def == typeHidden && sharedQueue[itr].second.def == typeSquash){
                    neuroSquash(Vec.array[hiddenOffset + sharedQueue[itr].first.id*ind]);
                }

                else if(sharedQueue[itr].first.def == typeMemGateIn && sharedQueue[itr].second.def == typeSquash){
                    neuroSquash(Vec.array[memGateInOffset + sharedQueue[itr].first.id*ind]);
                }

                else if(sharedQueue[itr].first.def == typeMemGateOut && sharedQueue[itr].second.def == typeSquash){
                    neuroSquash(Vec.array[memGateOutOffset + sharedQueue[itr].first.id*ind]);
                }

                else if(sharedQueue[itr].first.def == typeMemGateForget && sharedQueue[itr].second.def == typeSquash){
                    neuroSquash(Vec.array[memGateForgetOffset + sharedQueue[itr].first.id*ind]);
                }

                else if(sharedQueue[itr].first.def == typeOutput && sharedQueue[itr].second.def == typeSquash){
                    neuroSquash(Vec.array[outputOffset + sharedQueue[itr].first.id*ind]);
                }

            }

            Vec.array[whenMinOffset+j*ind] += shift(Vec.array[outputOffset+0*ind], 2160, 0); // nv = ((ov - omin)*(nmax-nmin) / (omax - omin))+nmin
            Vec.array[whenMaxOffset+j*ind] += shift(Vec.array[outputOffset+1*ind], 2160, 0);
            Vec.array[howCertainOffset+j*ind] += Vec.array[outputOffset+2*ind];
            Vec.array[communityMagOffset+j*ind] =  Vec.array[outputOffset+3*ind]; // set the next sets communityMag = output #3.
        }
    }
    for(int j=0; j<params.array[23]; j++){ // now lets get the average when and howcertain values.
        Vec.array[whenMinOffset+j*ind] = Vec.array[whenMinOffset+j*ind]/trainingsize;
        Vec.array[whenMaxOffset+j*ind] = Vec.array[whenMaxOffset+j*ind]/trainingsize;
        Vec.array[howCertainOffset+j*ind] = Vec.array[howCertainOffset+j*ind]/trainingsize;
    }
    /*calculate score for this individual during this round, current scoring mechanism is - e^(-(abs(whenGuess-whenAns)+distToCorrectSite)), closer to 1 the better.   */
    double maxCertainty=0;
    double whenMinGuess=0;
    double whenMaxGuess=0;
    double guessLat=0;
    double guessLon=0;

    for(int j=0; j<params.array[23]; j++){
        if(Vec.array[howCertainOffset+j*ind] > maxCertainty){
            maxCertainty = Vec.array[howCertainOffset+j*ind];
            whenMinGuess = Vec.array[whenMinOffset+j*ind];
            whenMaxGuess = Vec.array[whenMaxOffset+j*ind];
            guessLat = siteData[j*2];
            guessLon = siteData[j*2+1];
        }
    }

    Vec.array[fitnessOffset] = scoreFunc(whenMinGuess, whenMaxGuess, whenAns, guessLat, guessLon, ansLat, ansLon); //we take the average beacuse consistency is more important than being really good at this particular hour.
}
