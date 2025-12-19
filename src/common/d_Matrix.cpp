#include "d_Matrix.h"
#include "../es_cuda/d_math.h"

d_Matrix::d_Matrix(): rowCount(0), colCount(0), device_data(nullptr){}

d_Matrix::d_Matrix(const int rows, const int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	// Note: Using cudaMallocManaged for unified memory access
	// This allows both CPU and GPU to access the same memory
	// NN_Spiral and NN_PredictRadian require managed memory for their workflows
	d_check(cudaMallocManaged(VOID_PTR(&device_data), memSize()));
}

d_Matrix::d_Matrix(const float *host_data, const int rows, const int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	d_check(cudaMalloc(VOID_PTR(&device_data), memSize()));
	d_check(cudaMemcpyAsync(device_data, host_data, memSize(), cudaMemcpyHostToDevice));
}
d_Matrix::d_Matrix(const d_Matrix& other):	rowCount(other.rowCount),
											colCount(other.colCount)	
{
	d_check(cudaMalloc(VOID_PTR(&device_data), other.memSize()));
	d_check(cudaMemcpyAsync(device_data, other.device_data, other.memSize(), cudaMemcpyDeviceToDevice));
}
d_Matrix& d_Matrix::operator=(const d_Matrix& other)
{
	if (this == &other)
		return *this;
	free();
	rowCount = other.rowCount;
	colCount = other.colCount;
	d_check(cudaMalloc(VOID_PTR(&device_data), other.memSize()));
	d_check(cudaMemcpyAsync(device_data, other.device_data, other.memSize(), cudaMemcpyDeviceToDevice));
	return *this;
}
d_Matrix::~d_Matrix() {
	free();
}
d_Matrix d_Matrix::serialize() {
	d_Matrix result = d_Matrix(*this);
	result.serializeInPlace();
	return result;
}
d_Matrix d_Matrix::serialize() const {
	d_Matrix result = d_Matrix(*this);
	result.serializeInPlace();
	return result;
}
void d_Matrix::serializeInPlace() {
	colCount = size();
	rowCount = 1;
}
void d_Matrix::setShape(const int rows, const int cols) {
	rowCount = rows;
	colCount = cols;
}
void d_Matrix::free() {
	if (device_data == nullptr) {
		return;  // Nothing to free
	}
	
	cudaPointerAttributes attr = {};
	cudaPointerGetAttributes(&attr, device_data);
	if (attr.devicePointer != nullptr && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
		d_check(cudaFree(device_data));
		device_data = nullptr;
	}
	else if (attr.hostPointer != nullptr && attr.type == cudaMemoryTypeHost) {
		d_check(cudaFreeHost(device_data));
		device_data = nullptr;
	}
}
