#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL
#include "benchmarking.h"

unsigned long long CPUTimer(unsigned long long start=0){
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}