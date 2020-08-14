#pragma once
#include "d_cudahelpers.h"
class d_Matrix{
public:
	d_Matrix();
	d_Matrix(int rows, int cols);
	d_Matrix(float *host_data, int rows, int cols_);
	~d_Matrix();
	float* d_data(){
		return device_data;
	}
	float* d_data() const {
		return device_data;
	}
	int rows(){
		return rowCount;
	}
	int cols(){
		return colCount;
	}
	int size(){
		return rowCount * colCount;
	}
	size_t memSize(){
		return size() * sizeof(float);
	}
	void free(){
		cudaFree(device_data);
	}
protected:
	int rowCount;
	int colCount;
	float* device_data;
};
