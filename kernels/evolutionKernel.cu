#include <kernelDefs.h>
#include <thrust/random.h>
//using
extern __constant__ int params[];
//end of using

__global__ void evolutionKern(kernelArray<double> Vec, int *childOffset, uint32_t in, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ind = params[10];
    const int parentsIndex = *childOffset-1;
    const int fitnessval = params[19] + device_offset;
    int you, partner, child;
    thrust::minstd_rand0 randEng;
    randEng.seed(idx*in+in);
    thrust::uniform_int_distribution<size_t> selectParent(0,parentsIndex);
    randEng.discard(idx+1);
    you = selectParent(randEng);
    const int your_wt = params[11] + you + device_offset;
    randEng.discard(idx+1);
    partner = selectParent(randEng);
    while(fabs(Vec.array[fitnessval + partner]-Vec.array[fitnessval + you]) < 0.0025){ // not allowed to share the same fitness.
        randEng.discard(idx+1);
        partner = selectParent(randEng);
    }

    const int partner_wt = params[11] + partner + device_offset; // set the weights location for the partner
    child = parentsIndex + idx; // num threads = num eligible children
    Vec.array[params[19] + child + device_offset] = -1;
    const int child_wt = params[11] + child + device_offset; // set the weights location for the child
    const int child_mem = params[14] + child + device_offset;
    float mut = 0.037;
    for(int i=0; i<params[1]; i++){//for each weight, lets determine how the weights of the child are altered.
        thrust::uniform_real_distribution<float> weightSpin(0, 10);
        randEng.discard(idx+1);
        double result = 0;
        float rng = weightSpin(randEng);
        int first =0, second = 4.8;
        if(rng >=first && rng < second){
            result = Vec.array[your_wt + i*ind]; // give the child your weight val
        }
         first = second + 4.8;
        if(rng >= second && rng < first){
            result = Vec.array[partner_wt+i*ind];
        }

         second = first + mut;
        if(rng >=first && rng <second){
            result = Vec.array[partner_wt+i*ind]*2;
        }
         first = second + mut;
        if(rng >=second && rng <first){
            result = Vec.array[your_wt+i*ind]*2;
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = Vec.array[your_wt+i*ind]/2;
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = Vec.array[partner_wt+i*ind]/2;
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = Vec.array[your_wt+i*ind]/2;
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = -Vec.array[your_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = -Vec.array[partner_wt+i*ind];
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = Vec.array[your_wt+i*ind]-Vec.array[partner_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = Vec.array[partner_wt+i*ind]-Vec.array[your_wt+i*ind];
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = Vec.array[partner_wt+i*ind]+Vec.array[your_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <=second){
            result = Vec.array[your_wt+i*ind]+Vec.array[partner_wt+i*ind];
        }
        Vec.array[child_wt +i*ind] = result;
    }
    for(int i=0; i<params[5]; i++){
        Vec.array[child_mem + i*ind] =0;
    }
    for(int i=0; i<params[23]; i++){
        Vec.array[params[20] + i*ind] = 0;
    }
}
