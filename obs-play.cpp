// Playing with OBS plugins..
//
//
#include <obs-module.h>
#include <stdio.h>

OBS_DECLARE_MODULE()

bool obs_module_load(void) {
    puts("Hello Mum!");
    return true;
}
