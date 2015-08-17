#ifndef SYSINFO_H
#define SYSINFO_H
#include <sys/sysinfo.h>
#include <cuda_runtime_api.h>


size_t GetHostRamInBytes();


size_t GetDeviceRamInBytes();

#endif
