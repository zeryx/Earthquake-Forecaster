#include <kernelDefs.h>
#include <thrust/random.h>


__global__ void evolutionKern(kernelArray<float> Vec, kernelArray<int> params, int *childOffset, int *realGridSize, size_t in, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < *realGridSize){
        const int ind = params.array[10];
        const int parentsIndex = *childOffset-1;
        const int fitnessval = params.array[19] + device_offset;
        int you, partner, child;
        thrust::minstd_rand0 randEng(in);
        thrust::uniform_int_distribution<int> selectParent(0,parentsIndex);
        randEng.discard(idx+20);
        you = selectParent(randEng);
        const int your_wt = params.array[11] + you + device_offset;
        randEng.discard(idx+20);
        partner = selectParent(randEng);
        while(Vec.array[fitnessval + partner] == Vec.array[fitnessval + you]){ // not allowed to share the same fitness.
            randEng.discard(idx+20);
            partner = selectParent(randEng);
        }
        const int partner_wt = params.array[11] + partner + device_offset; // set the weights location for the partner
        child = parentsIndex + idx; // num threads = num eligible children

        const int child_wt = params.array[11] + child + device_offset; // set the weights location for the child
        const int child_mem = params.array[14] + child + device_offset;
        const int child_communityMag = params.array[20] + child + device_offset;

        float mut = 0.7692;
        float chance[14];
        chance[0] = 0;
        for(int i=1; i<14; i++){
            chance[i] = chance[i-1] + mut;
        }
        //reset parameters for this child.

        for(int i=0; i<params.array[5]; i++){
            Vec.array[child_mem + i*ind] =0;
        }
        for(int i=0; i<params.array[23]; i++){
            Vec.array[child_communityMag + i*ind] = 0;
        }

        for(int i=0; i<params.array[1]; i++){//for each weight, lets determine how the weights of the child are altered.
            thrust::uniform_real_distribution<float> weightSpin(0, 10);
            randEng.discard(idx+4);
            float result;
            float rng = weightSpin(randEng);

            if(rng >=chance[0] && rng<chance[1])
                result = Vec.array[partner_wt+i*ind]*2;


            else if(rng >=chance[1] && rng<chance[2])
                result = Vec.array[your_wt+i*ind]*2;


            else if(rng>=chance[2] && rng<chance[3])
                result = Vec.array[your_wt+i*ind]/2;


            else if(rng>=chance[3] && rng<chance[4])
                result = Vec.array[partner_wt+i*ind]/2;


            else if(rng>=chance[4] && rng<chance[5])
                result = Vec.array[your_wt+i*ind]/2;


            else if(rng>=chance[5] && rng<chance[6])
                result = -Vec.array[your_wt+i*ind];


            else if(rng>=chance[6] && rng<chance[7])
                result = -Vec.array[partner_wt+i*ind];


            else if(rng>=chance[7] && rng<chance[8])
                result = Vec.array[your_wt+i*ind]-Vec.array[partner_wt+i*ind];


            else if(rng>=chance[8] && rng<chance[9])
                result = Vec.array[partner_wt+i*ind]-Vec.array[your_wt+i*ind];


            else if(rng>=chance[9] && rng<chance[10])
                result = Vec.array[partner_wt+i*ind]+Vec.array[your_wt+i*ind];


            else if(rng>=chance[10] && rng<chance[11])
                result = Vec.array[your_wt+i*ind]+Vec.array[partner_wt+i*ind];
            else if(rng>=chance[11] && rng<chance[12])
                result = Vec.array[your_wt+i*ind];
            else if(rng>=chance[12] && rng<chance[13])
                result = Vec.array[partner_wt+i*ind];
            else // this should almost never be the set value.
                result =10;

            Vec.array[child_wt +i*ind] = result;
        }
    }
}
