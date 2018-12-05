#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "types.h"
#include "color.h"

void d_add(float *c, const float *a, const float *b, unsigned int size);
void d_subtract(float *c, const float *a, const float *b, unsigned int size);
void d_drawPixels(int * buffer, int m, int k, const float * vals, bool discrete);
void d_forwardLayer(float* d_Z, const float* d_last, const float* d_W, const float* d_b, int midDim, int lfsRows, int rhsCols);
void d_Activate(float* dst, int size, Activation act);
void d_BackTanh(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k);
void d_BackReLU(float *dst, float *d_W, float *d_dZ, const float *d_A, int m, int n, int k);
void d_matrixMult(float* dst, const float* d_W, const float* d_last, int  m, int n, int k);