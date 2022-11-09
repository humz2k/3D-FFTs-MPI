#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

extern "C" {
    unsigned long long CPUTimer(unsigned long long start){

        timeval tv;
        gettimeofday(&tv, 0);
        return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;

    }
}