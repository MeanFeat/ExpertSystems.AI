#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "windows.h"
#include <iostream>

/**
 * CUDA error checking and profiling utilities
 * Provides macros for debugging CUDA operations and measuring performance
 */

#ifdef _DEBUG
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

/**
 * Helper function to stop CUDA profiling and calculate elapsed time
 */
inline void profileStop(cudaEvent_t start, cudaEvent_t stop, float *output) {
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(output, start, stop);
}

/**
 * Check CUDA error and output debug information if error occurred
 * @param err CUDA error code to check
 * @param file Source file where error check is performed
 * @param line Line number where error check is performed
 */
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

/**
 * Catch and report the last CUDA error
 * @param file Source file where error catch is performed
 * @param line Line number where error catch is performed
 */
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