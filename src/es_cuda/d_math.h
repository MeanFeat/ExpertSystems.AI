#pragma once
#include "d_cudahelpers.h"
#include "types.h"
#include "d_Matrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

extern bool isInitialized;
extern cublasHandle_t cublasHandle;

typedef unsigned int uint;

#define BLOCK_SIZE 16
#define LRELU_LEAK 0.01f
#define VOID_PTR(t) reinterpret_cast<void**>(t)
using ptrFunc = float(*)(float, float);

void d_mathInit();

inline dim3 dimGrid(uint m, uint k) {
	return dim3((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
inline dim3 dimBlock() {
	return dim3(BLOCK_SIZE, BLOCK_SIZE);
}
/* dst (=) a */
void d_set_elem(float *dst, const float a);
/* dst (=) a */
void d_set_elem(d_Matrix *dst, const float a);
/* dst = srcA (+) srcB */
void d_add_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB);
/* dst = srcA (-) srcB */
void d_subtract_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB);
/* dst = srcA (*) srcB */
void d_mult_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB);
/* dst = srcA * b */
void d_mult_scalar(d_Matrix *dst, const float b);
/* dst = src.T */
void d_transpose(d_Matrix *dst, const d_Matrix *src);
/* dst = srcA * srcB */
void d_mult(d_Matrix * dst, const d_Matrix * srcA, const d_Matrix * srcB);
/* dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix * dst, const d_Matrix * srcA, const d_Matrix * srcB);
/* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix * dst, const d_Matrix * srcA, const d_Matrix * srcB);
void d_sumMatrix(float* dst, const d_Matrix* src);
void d_square(d_Matrix* dst, const d_Matrix* src);
void d_forwardLayer(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_last, const d_Matrix *d_bias);
void d_activate(d_Matrix *dst, Activation act);
void d_backSigmoid(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A);
void d_backTanh(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A);
void d_backReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A);
void d_backLReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A);
void d_backSine(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A);
void d_set_dW(d_Matrix *dst, const d_Matrix *d_dZ, const d_Matrix *d_A, float coefficient);
void d_set_dW_Reg(d_Matrix *dst, const d_Matrix *d_dZ, const d_Matrix *d_A, const d_Matrix *d_W, float coefficient, float regTerm);
void d_sumRows(d_Matrix* dst, const d_Matrix* src);
void d_set_db(d_Matrix *dst, const d_Matrix *d_dZ, float coefficient);
void d_updateParameterADAM(d_Matrix * dst, const d_Matrix *d_derivative, const d_Matrix *d_momentum, const d_Matrix *d_momentumSqr, float learnRate);
void d_updateParameter(d_Matrix * dst, const d_Matrix * d_derivative, float learnRate);
void d_calcCost(float *dst, float *weightSum, float *epochCost, const d_Matrix* d_err, const std::vector<d_Matrix>* d_modelWeights, float regMult, float coeff, float trainLabelCount);
void d_averageCost(float *epochCost, int batchCount);
void d_gatherColumns(float *dst, const float *src, const int *indices, int rows, int count);
void d_drawPixels(int * buffer, uint m, uint k, const float *vals, bool discrete);

inline __device__ float _set_elem(float a, const float b) {
	return b;
}