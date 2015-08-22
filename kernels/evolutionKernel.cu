#include <kernelDefs.h>
#include <thrust/random.h>


__global__ void evoFirstKern(kernelArray<double> vect, kernelArray<int> params, float avgFitness,  int device_offset){//population new/old = params[25]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // for each thread is one individual
    int fitnessval = params.array[19] + idx + device_offset;
    vect.array[fitnessval] = vect.array[fitnessval]/avgFitness;
}

__global__ void evoSecondKern(  kernelArray<double> vect, kernelArray<int> params, int device_offset, Lock lock){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int yourFitness = params.array[19] + idx + device_offset;
    if(vect.array[yourFitness] >= 1){
        int partner, child;
        const int your_wt = params.array[11] + idx + device_offset;
        thrust::minstd_rand0 randEng;
        thrust::uniform_int_distribution<int> select(0,params.array[10]);
        while(1){//select partner
            randEng.discard(vect.array[yourFitness]);
            if(vect.array[params.array[19] + select(randEng) + device_offset] >= 1){
                partner = select(randEng);
                break;
            }
        }
        const int partner_wt = params.array[11] + partner + device_offset; // set the weights location for the partner

        while(1){//select child, use locks.
            randEng.discard(vect.array[yourFitness]);
            lock.lock();
            if(vect.array[params.array[19]+select(randEng) + device_offset] <0 && vect.array[params.array[19]+select(randEng)+device_offset] != -1){
                vect.array[params.array[19]+select(randEng)+device_offset] = -1;
                child = select(randEng);
                lock.unlock();
                break;
            }
        }
        const int child_wt = params.array[11] + child + device_offset; // set the weights location for the child

        for(int i=0; i<params.array[1]; i++){//for each weight, lets determine how the weights of the child are altered.
            thrust::uniform_real_distribution<float> weightSpin(0, 10);
            randEng.discard(vect.array[yourFitness]);
            float rng = weightSpin(randEng);
            if(rng >=0 && rng < 3){
                vect.array[child_wt + i] = vect.array[your_wt + i]; // give the child your weight val
            }
            else if(rng >=3 && rng <6){
                vect.array[child_wt+i] = vect.array[partner_wt+i]; // the the child your partners val
            }
            else if(rng >=6 && rng <8){ //less of a chance, 20%
                vect.array[child_wt+i] = vect.array[child_wt+i]; // the child keeps the dead individuals weight
            }
            else if(rng >=8 && rng <8.5){// 5% chance for this mutation
                mutations your_mt, partner_mt, child_mt;
                your_mt.result = vect.array[your_wt+i];
                partner_mt.result = vect.array[partner_wt+i];
                child_mt.f[0] = your_mt.f[1];
                child_mt.f[1] = partner_mt.f[1];
                vect.array[child_wt+i] = child_mt.result;
            }
            else if(rng >=8.5 && rng <9){ //same last, but positions reversed.
                mutations your_mt, partner_mt, child_mt;
                your_mt.result = vect.array[your_wt+i];
                partner_mt.result = vect.array[partner_wt+i];
                child_mt.f[1] = your_mt.f[1];
                child_mt.f[0] = partner_mt.f[1];
                vect.array[child_wt+i] = child_mt.result;
            }
            else if(rng >= 9 && rng < 9.5){ //this takes the last 4 bits rather than the first 4.
                mutations your_mt, partner_mt, child_mt;
                your_mt.result = vect.array[your_wt+i];
                partner_mt.result = vect.array[partner_wt+i];
                child_mt.f[1] = your_mt.f[0];
                child_mt.f[0] = partner_mt.f[0];
                vect.array[child_wt+i] = child_mt.result;
            }
            else if(rng >=9.5 && rng <= 10){//similar as the last two but more bytes being shuffled around
                mutations your_mt, partner_mt, child_mt;
                your_mt.result = vect.array[your_wt+i];
                partner_mt.result = vect.array[partner_wt+i];
                child_mt.c[0] = your_mt.c[0];
                child_mt.c[1] = partner_mt.c[0];
                child_mt.c[2] = your_mt.c[1];
                child_mt.c[3] = partner_mt.c[1];
                child_mt.c[4] = your_mt.c[2];
                child_mt.c[5] = partner_mt.c[2];
                child_mt.c[6] = your_mt.c[3];
                child_mt.c[7] = partner_mt.c[3];
                vect.array[child_wt+i] = child_mt.result;
            }
        }
    }
}
