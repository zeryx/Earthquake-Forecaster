#include <kernelDefs.h>
#include <thrust/random.h>


__global__ void evolutionKern(kernelArray<double> vect, kernelArray<int> params, uint32_t in, size_t device_offset){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ind = params.array[10];
    int you, partner, child;
    thrust::minstd_rand0 randEng;
    thrust::uniform_int_distribution<size_t> select(0,params.array[10]);
    while(1){//select primary
        randEng.discard(idx+in);
        if(vect.array[params.array[19] + select(randEng) + device_offset] >0){//everyone below 1.15 was already deleted
            you = select(randEng);
            break;
        }
    }
    const int your_wt = params.array[11] + you + device_offset;
    while(1){//select secondary
        randEng.discard(idx+in);
        if(vect.array[params.array[19] + select(randEng) + device_offset] > 0){ //dido as before, the eligible parent value is set in normalize
            partner = select(randEng);
            break;
        }
    }
    const int partner_wt = params.array[11] + partner + device_offset; // set the weights location for the partner

    while(1){//select child.
        randEng.discard(in + idx);
        if(vect.array[params.array[19] + select(randEng) + device_offset] == 0){
            vect.array[params.array[19] + select(randEng) + device_offset] = -1;
            child = select(randEng);
            break;
        }
    }
    const int child_wt = params.array[11] + child + device_offset; // set the weights location for the child
    const int child_mem = params.array[14] + child + device_offset;

    for(int i=0; i<params.array[1]; i++){//for each weight, lets determine how the weights of the child are altered.
        mutations your_mt, partner_mt, child_mt;
        your_mt.result = vect.array[your_wt+i*ind];
        partner_mt.result = vect.array[partner_wt+i*ind];

        thrust::uniform_real_distribution<float> weightSpin(0, 10);
        randEng.discard(in + idx);
        double result;
        float rng = weightSpin(randEng);

        if(rng >=0 && rng < 5){ //50%
            result = vect.array[your_wt + i*ind]; // give the child your weight val
        }
        else if(rng >=5 && rng <9){ //40%
            result = vect.array[partner_wt+i*ind]; // the the child your partners val
        }
        else if(rng >=9 && rng <9.9){ //less of a chance, 9.9%
            result = vect.array[child_wt+i*ind]; // the child keeps the dead individuals weight
        }
        else if(rng >=9.9 && rng <9.95){ //0.05% chance of this mutation
            child_mt.f[0] = your_mt.f[1];
            child_mt.f[1] = partner_mt.f[1];
            result = child_mt.result;
        }
        else if(rng >=9.95 && rng <= 10){ //0.05% chance of this mutation
            child_mt.f[1] = your_mt.f[1];
            child_mt.f[0] = partner_mt.f[1];
            result = child_mt.result;
        }
        //            else if(rng >= 9.75 && rng < 9.90){ //this takes the last 4 bits rather than the first 4.
        //                child_mt.f[1] = your_mt.f[0];
        //                child_mt.f[0] = partner_mt.f[0];
        //                result = child_mt.result;
        //            }
        //            else if(rng >=9.9 && rng <= 10){//similar as the last two but more bytes being shuffled around
        //                child_mt.c[0] = your_mt.c[0];
        //                child_mt.c[1] = partner_mt.c[0];
        //                child_mt.c[2] = your_mt.c[1];
        //                child_mt.c[3] = partner_mt.c[1];
        //                child_mt.c[4] = your_mt.c[2];
        //                child_mt.c[5] = partner_mt.c[2];
        //                child_mt.c[6] = your_mt.c[3];
        //                child_mt.c[7] = partner_mt.c[3];
        //                result = child_mt.result;
        //            }
        vect.array[child_wt +i*ind] = result;
    }
    for(int i=0; i<params.array[5]; i++){
        vect.array[child_mem + i*ind] =0;
    }
}
