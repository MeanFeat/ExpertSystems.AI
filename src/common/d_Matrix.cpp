#include "d_Matrix.h"
#include "..\es_cuda\d_math.h"
d_Matrix::d_Matrix(): rowCount(0), colCount(0), device_data(nullptr) {}
d_Matrix::d_Matrix(const int rows, const int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	device_data = nullptr;
	if (memSize() > 0) d_check(cudaMalloc(VOID_PTR(&device_data), memSize()));
}
d_Matrix::d_Matrix(const float *host_data, const int rows, const int cols) {
	this->rowCount = rows;
	this->colCount = cols;
	device_data = nullptr;
	if (memSize() > 0) {
		d_check(cudaMalloc(VOID_PTR(&device_data), memSize()));
		d_check(cudaMemcpy(device_data, host_data, memSize(), cudaMemcpyHostToDevice));
	}
}
d_Matrix::d_Matrix(const d_Matrix& other):	rowCount(other.rowCount),
											colCount(other.colCount),
											device_data(nullptr)
{
	if (other.memSize() > 0) {
		d_check(cudaMalloc(VOID_PTR(&device_data), other.memSize()));
		d_check(cudaMemcpy(device_data, other.device_data, other.memSize(), cudaMemcpyDeviceToDevice));
	}
}
d_Matrix::d_Matrix(d_Matrix&& other) noexcept : rowCount(other.rowCount), colCount(other.colCount), device_data(other.device_data) {
	other.rowCount = 0;
	other.colCount = 0;
	other.device_data = nullptr;
}
d_Matrix& d_Matrix::operator=(const d_Matrix& other)
{
	if (this == &other)
		return *this;
	free();
	rowCount = other.rowCount;
	colCount = other.colCount;
	if (other.memSize() > 0) {
		d_check(cudaMalloc(VOID_PTR(&device_data), other.memSize()));
		d_check(cudaMemcpy(device_data, other.device_data, other.memSize(), cudaMemcpyDeviceToDevice));
	}
	return *this;
}
d_Matrix& d_Matrix::operator=(d_Matrix&& other) noexcept {
	if (this == &other) return *this;
	free();
	rowCount = other.rowCount;
	colCount = other.colCount;
	device_data = other.device_data;
	other.rowCount = 0;
	other.colCount = 0;
	other.device_data = nullptr;
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
	if (device_data) {
		d_check(cudaFree(device_data));
		device_data = nullptr;
	}
}
