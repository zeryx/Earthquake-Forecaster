#ifndef SYSINFO_H
#define SYSINFO_H
#include <sys/sysinfo.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
size_t GetHostRamInBytes(void)
{
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if(meminfo == NULL)
        exit(1);

    char line[256];
    while(fgets(line, sizeof(line), meminfo))
    {
        size_t  ram;
        if(sscanf(line, "MemFree: %zu kB", &ram) == 1)
        {
            fclose(meminfo);
            return ram*1000;
        }
    }

    // If we got here, then we couldn't find the proper line in the meminfo file:
    // do something appropriate like return an error code, throw an exception, etc.
    fclose(meminfo);
    std::cerr<<"cannot find line in meminfo file, please doublecheck."<<std::endl;
    exit(1);
}

size_t GetDeviceRamInBytes(void){
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

#endif
