#include <kernelDefs.h>
#include <thrust/random.h>


__global__ void evolutionKern(kernelArray<double> vect, kernelArray<int> params, int *childOffset, uint32_t in, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ind = params.array[10];
    const int parentsIndex = *childOffset-1;
    const int fitnessval = params.array[19] + device_offset;
    int you, partner, child;
    thrust::minstd_rand0 randEng;
    randEng.seed(idx*in+in);
    thrust::uniform_int_distribution<size_t> selectParent(0,parentsIndex);
    randEng.discard(idx+1);
    you = selectParent(randEng);
    const int your_wt = params.array[11] + you + device_offset;
    randEng.discard(idx+1);
    partner = selectParent(randEng);
    while(fabs(vect.array[fitnessval + partner]-vect.array[fitnessval + you]) < 0.0025){ // not allowed to share the same fitness.
        randEng.discard(idx+1);
        partner = selectParent(randEng);
    }

    const int partner_wt = params.array[11] + partner + device_offset; // set the weights location for the partner
    child = parentsIndex + idx; // num threads = num eligible children
    vect.array[params.array[19] + child + device_offset] = -1;
    const int child_wt = params.array[11] + child + device_offset; // set the weights location for the child
    const int child_mem = params.array[14] + child + device_offset;
    float mut = 0.037;
    for(int i=0; i<params.array[1]; i++){//for each weight, lets determine how the weights of the child are altered.
        thrust::uniform_real_distribution<float> weightSpin(0, 10);
        randEng.discard(idx+1);
        double result = 0;
        float rng = weightSpin(randEng);
        int first =0, second = 4.8;
        if(rng >=first && rng < second){
            result = vect.array[your_wt + i*ind]; // give the child your weight val
        }
         first = second + 4.8;
        if(rng >= second && rng < first){
            result = vect.array[partner_wt+i*ind];
        }

         second = first + mut;
        if(rng >=first && rng <second){
            result = vect.array[partner_wt+i*ind]*2;
        }
         first = second + mut;
        if(rng >=second && rng <first){
            result = vect.array[your_wt+i*ind]*2;
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = vect.array[your_wt+i*ind]/2;
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = vect.array[partner_wt+i*ind]/2;
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = vect.array[your_wt+i*ind]/2;
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = -vect.array[your_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = -vect.array[partner_wt+i*ind];
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = vect.array[your_wt+i*ind]-vect.array[partner_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <second){
            result = vect.array[partner_wt+i*ind]-vect.array[your_wt+i*ind];
        }
          first = second + mut;
        if(rng >=second && rng <first){
            result = vect.array[partner_wt+i*ind]+vect.array[your_wt+i*ind];
        }
          second = first + mut;
        if(rng >=first && rng <=second){
            result = vect.array[your_wt+i*ind]+vect.array[partner_wt+i*ind];
        }
        vect.array[child_wt +i*ind] = result;
    }
    for(int i=0; i<params.array[5]; i++){
        vect.array[child_mem + i*ind] =0;
    }
    for(int i=0; i<params.array[23]; i++){
        vect.array[params.array[20] + i*ind] = 0;
    }
}
