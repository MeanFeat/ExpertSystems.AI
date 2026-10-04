#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "windows.h"
#include <iostream>

#if defined(_DEBUG) || defined(ES_CUDA_CHECKS)
#define CUDA_ERROR_CHECK
#define d_check( err ) checkErr( err, __FILE__, __LINE__ )
#define d_catchErr()    catchErr( __FILE__, __LINE__ )
#else // _RELEASE
#define d_check( err ) err
#define d_catchErr()
#endif

#ifdef _PROFILE
#define d_profile(start,stop,output, args ) cudaEventRecord(start); args; profileStop(start,stop,output)
#else
#define d_profile(start,stop,output, args ) args;
#endif

inline void profileStop(cudaEvent_t start, cudaEvent_t stop, float *output) {
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(output, start, stop);
}
inline void checkErr(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		char txt[256] = { 0 };
		_snprintf_s(txt, 256, "error : #CUDA ERROR::checkErr() failed at %s (%i) : %s\n", file, line, cudaGetErrorString(err));
		OutputDebugStringA(txt);
		exit(-1);
	}
#endif
	return;
}
inline void catchErr(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		char txt[256] = { 0 };
		_snprintf_s(txt, 256, "error : #CUDA ERROR::catchErr() failed at %s (%i) : %s\n", file, line, cudaGetErrorString(err));
		OutputDebugStringA(txt);
		exit(-1);
	}
#endif
	return;
}