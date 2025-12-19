#pragma once
#include "d_cudahelpers.h"

/**
 * CUDA device matrix class for GPU-accelerated matrix operations
 * Manages memory allocation and deallocation on the GPU
 */
class d_Matrix {
public:
	d_Matrix();
	d_Matrix(int rows, int cols);
	d_Matrix(const float *host_data, int rows, int cols);
	d_Matrix(const d_Matrix& other);
	d_Matrix& operator=(const d_Matrix& other);
	~d_Matrix();
	float* &d_data() { return device_data; }
	const float* d_data() const { return device_data; }
	int rows() const { return rowCount; }
	int cols() const { return colCount; }
	int size() const { return rowCount * colCount; }
	d_Matrix serialize();
	d_Matrix serialize() const;
	void serializeInPlace();
	void setShape(int rows, int cols);
	size_t memSize() { return size() * sizeof(float); }
	size_t memSize() const { return size() * sizeof(float); }
	void free();
private:
	int rowCount;
	int colCount;
	float* device_data;
};
