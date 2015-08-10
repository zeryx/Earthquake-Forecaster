#ifndef MEMMANAGER_H
#define MEMMANAGER_H
#include "stdlib.h"

namespace memManager{

void* alloc(size_t len);

void dealloc(void* ptr);

}
#endif
