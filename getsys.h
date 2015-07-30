#ifndef DATAARRAY_H
#define DATAARRAY_H
#include <sys/sysinfo.h>

#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
int GetHostRamInBytes(void)
{
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if(meminfo == NULL)
        return 0;

    char line[256];
    while(fgets(line, sizeof(line), meminfo))
    {
        int ram;
        if(sscanf(line, "memFree: %d kB", &ram) == 1)
        {
            fclose(meminfo);
            return ram*1000;
        }
    }

    // If we got here, then we couldn't find the proper line in the meminfo file:
    // do something appropriate like return an error code, throw an exception, etc.
    fclose(meminfo);
    return -1;
}


int GetDeviceRamInBytes(void){
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return (int) free;
}

#endif
