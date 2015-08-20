#include <kernelDefs.h>

 __constant__ int input[15*3*35]{0};
 __constant__ int site_offset[15]{0};//padding
 __constant__ int channel_offset[3];
 __constant__ int trainingsize;
