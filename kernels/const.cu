#include <kernelDefs.h>

 __constant__ int input[20*3*40];
 __constant__ int site_offset[20];
 __constant__ int channel_offset[3];
 __constant__ int trainingsize;
