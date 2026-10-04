#include "d_math.h"
#include "color.h"
using namespace std;
bool isInitialized = false;
cublasHandle_t cublasHandle;
__device__ ptrFunc d_pfAdd = __fadd_rn;
__device__ ptrFunc d_pfSub = __fsub_rn;
__device__ ptrFunc d_pfMult = __fmul_rn;
__device__ ptrFunc d_pfSet = _set_elem;
namespace {
	ptrFunc pfAdd;
	ptrFunc pfSub;
	ptrFunc pfMult;
	ptrFunc pfSet;
	void checkCublas(cublasStatus_t status) {
		if (status != CUBLAS_STATUS_SUCCESS) {
			OutputDebugStringA("cuBLAS initialization failed.\n");
			exit(EXIT_FAILURE);
		}
	}
}
#define setFunctionPointer(h_ptr, d_ptr) cudaMemcpyFromSymbol(&(h_ptr), d_ptr, sizeof(ptrFunc));
__device__
uint GetRow(){
	return blockIdx.y * blockDim.y + threadIdx.y;
}
__device__
uint GetCol() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}
void d_mathInit() {
	if (!isInitialized) {
		checkCublas(cublasCreate(&cublasHandle));
		setFunctionPointer(pfAdd, d_pfAdd)
		setFunctionPointer(pfSub, d_pfSub)
		setFunctionPointer(pfMult, d_pfMult)
		setFunctionPointer(pfSet, d_pfSet)
		isInitialized = true;
	}
}
__global__
void launch2D_elem_Kernel(const ptrFunc op, float * dst, const float *a, const float *b, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = (*op)(a[tid], b[tid]);
	}
}
__global__
void launch_elem_broad_Kernel(const ptrFunc op, float * dst, const float *a, const float b, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = (*op)(a[tid], b);
	}
}
/* dst (op)= a */
void d_launch_single_thread(const ptrFunc op, float *dst, const float a) {
	launch_elem_broad_Kernel << <1, 1 >> > (op, dst, dst, a, 1, 1);
}
/* dst (op)= srcA */
void d_launch_single_thread(const ptrFunc op, float *dst, const float *srcA) {
	launch2D_elem_Kernel << <1, 1 >> > (op, dst, dst, srcA, 1, 1);
}
/* dst = srcA (op) b */
void d_launch_single_thread(const ptrFunc op, float *dst, const float *srcA, const float b) {
	launch_elem_broad_Kernel << <1, 1 >> > (op, dst, srcA, b, 1, 1);
}
/* dst = srcA (op) srcB */
void d_launch_single_thread(const ptrFunc op, float *dst, const float *srcA, const float *srcB) {
	launch2D_elem_Kernel << <1, 1 >> > (op, dst, srcA, srcB, 1, 1);
}
/* dst = srcA (op) srcB */
void d_launch2D_elem(const ptrFunc func, d_Matrix *dst, const float *srcA, const float *srcB) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	launch2D_elem_Kernel << <dimGrid(m, k), dimBlock() >> > (func, dst->d_data(), srcA, srcB, m, k);
	d_catchErr();
}
/* dst = srcA (op) b */
void d_launch2D_elem(const ptrFunc func, d_Matrix *dst, const float * srcA, const float b) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	launch_elem_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (func, dst->d_data(), srcA, b, m, k);
	d_catchErr();
}
/* dst = a */
void d_set_elem(float *dst, const float a) {
	d_launch_single_thread(pfSet, dst, a);
	d_catchErr();
}
/* dst = a */
void d_set_elem(d_Matrix *dst, const float a) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	launch_elem_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (pfSet, dst->d_data(), dst->d_data(), a, m, k);
	d_catchErr();
}
/* dst = srcA (+) srcB */
void d_add_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfAdd, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (-) srcB */
void d_subtract_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfSub, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
/* dst = srcA (*) srcB */
void d_mult_elem(d_Matrix *dst, const d_Matrix &srcA, const d_Matrix &srcB) {
	d_launch2D_elem(pfMult, dst, srcA.d_data(), srcB.d_data());
	d_catchErr();
}
__global__
void mult_scalar_Kernel(float *dst, const float b, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = dst[tid] * b;
	}
}
void d_mult_scalar(float *dst, const float b, const uint m, const uint k) {
	mult_scalar_Kernel << <dimGrid(m, k), dimBlock() >> > (dst, b, m, k);
	d_catchErr();
}
void d_mult_scalar(d_Matrix *dst, const float b) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	d_mult_scalar(dst->d_data(), b, m, k);
	d_catchErr();
}
__global__
void transpose_Kernel(float *dst, const float *src, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (row < m && col < k) {
		dst[col + row * k] = src[row + col * m];
	}
} /* dst = src.T */
void d_transpose(d_Matrix *dst, const d_Matrix *src) {
	const int c = src->cols();
	const int r = src->rows();
	constexpr float alpha = 1.f;
	constexpr float beta = 0.f;
	cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
		c, r, &alpha,
		src->d_data(), r,
		&beta,
		src->d_data(), r,
		dst->d_data(), c);
	d_catchErr();
} /* dst = srcA * srcB */
void d_mult(d_Matrix* dst, const d_Matrix* srcA, const d_Matrix* srcB) {
	constexpr float alpha = 1.f;
	constexpr float beta = 0.f;
	const int m = srcA->rows();
	const int n = srcB->cols();
	const int k = srcA->cols();
	cublasSgemm(cublasHandle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		srcA->d_data(), m,
		srcB->d_data(), k,
		&beta,
		dst->d_data(), m);
	d_catchErr();
} /*dst = srcA.T * srcB */
void d_mult_lhsT(d_Matrix* dst, const d_Matrix* srcA, const d_Matrix* srcB) {
	constexpr float alpha = 1.f;
	constexpr float beta = 0.f;
	const int m = srcA->cols();
	const int n = srcB->cols();
	const int k = srcA->rows();
	cublasSgemm(cublasHandle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		srcA->d_data(), k,
		srcB->d_data(), k,
		&beta,
		dst->d_data(), m);
	d_catchErr();
} /* dst = srcA * srcB.T */
void d_mult_rhsT(d_Matrix* dst, const d_Matrix *srcA, const d_Matrix *srcB) {
	constexpr float alpha = 1.f;
	constexpr float beta = 0.f;
	const int m = srcA->rows();
	const int n = srcB->rows();
	const int k = srcB->cols();
	cublasSgemm(cublasHandle,
		CUBLAS_OP_N, CUBLAS_OP_T,
		m, n, k,
		&alpha,
		srcA->d_data(), m,
		srcB->d_data(), n,
		&beta,
		dst->d_data(), m);
	d_catchErr();
}
__global__
void sumValues_Kernel(float *dst, const float *src, const uint count) {
	__shared__ float partial[256];
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0.f;
	for (uint i = tid; i < count; i += blockDim.x * gridDim.x) {
		value += src[i];
	}
	partial[threadIdx.x] = value;
	__syncthreads();
	for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(dst, partial[0]);
}
__global__
void sumSquares_Kernel(float *dst, const float *src, const uint count) {
	__shared__ float partial[256];
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0.f;
	for (uint i = tid; i < count; i += blockDim.x * gridDim.x) {
		value += src[i] * src[i];
	}
	partial[threadIdx.x] = value;
	__syncthreads();
	for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	if (threadIdx.x == 0) atomicAdd(dst, partial[0]);
}
__global__
void gatherColumns_Kernel(float *dst, const float *src, const int *indices, const uint rows, const uint count) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	const uint total = rows * count;
	if (tid < total) {
		const uint row = tid % rows;
		const uint col = tid / rows;
		dst[tid] = src[uint(indices[col]) * rows + row];
	}
}
void d_sumMatrix(float* dst, const d_Matrix *src) {
	d_Matrix r = d_Matrix(src->rows(), 1);
	d_sumRows(&r, src); d_catchErr();
	r.setShape(1, r.rows());
	d_sumRows(&r, &r); d_catchErr();
	cudaMemcpyAsync(VOID_PTR(dst), r.d_data(), sizeof(float), cudaMemcpyDeviceToDevice); d_catchErr();
}
__global__
void add_row_broad_Kernel(float *dst, const float *srcMat, const float *srcVec, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		const uint index = col * m + row;
		dst[index] = __fadd_rd(srcMat[index], srcVec[row]);
	}
}
/* dst = d_W * d_last + d_bias */
void d_forwardLayer(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_last, const d_Matrix *d_bias) {
	const uint m = d_W->rows();
	const uint k = d_last->cols();
	d_mult(dst, d_W, d_last);
	add_row_broad_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), dst->d_data(), d_bias->d_data(), m, k);
	d_catchErr();
}
__global__
void drawPixels_Kernel(int *buffer, const uint m, const float* vals, const bool discrete, const Color neg, const Color pos) {
	const uint row = GetRow();
	const uint col = GetCol();
	const float percent = vals[col * m + row];
	if (discrete) {
		if (percent > 0.f) {
			buffer[col * m + row] = ((pos.r << 16) | ((pos.g << 8) | pos.b));
		}
		else {
			buffer[col * m + row] = ((neg.r << 16) | ((neg.g << 8) | neg.b));
		}
	}
	else {
		if (percent > 0.) {
			const unsigned char r = unsigned char(float(255) + (percent*(float(pos.r) - float(255))));
			const unsigned char g = unsigned char(float(255) + (percent*(float(pos.g) - float(255))));
			const unsigned char b = unsigned char(float(255) + (percent*(float(pos.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (percent*(float(pos.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
		else {
			const unsigned char r = unsigned char(float(255) + (-percent * (float(neg.r) - float(255))));
			const unsigned char g = unsigned char(float(255) + (-percent * (float(neg.g) - float(255))));
			const unsigned char b = unsigned char(float(255) + (-percent * (float(neg.b) - float(255))));
			//unsigned char a = unsigned char(float(255) + (-percent*(float(neg.a) - float(255))));
			buffer[col * m + row] = ((r << 16) | ((g << 8) | b));
		}
	}
}
void d_drawPixels(int *buffer, const uint m, const uint k, const float* vals, const bool discrete) {
	const Color pos = Color(100, 167, 211, 255);
	const Color neg = Color(255, 184, 113, 255);
	drawPixels_Kernel << <dimGrid(m, k), dimBlock() >> >
		(buffer, m, vals, discrete, neg, pos);
	d_catchErr();
}
__global__
void Sigmoid_Kernel(float *dst, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = __fdividef(1.f, (__fadd_rd(1.f, __expf(-dst[tid]))));
	}
}
__global__
void Tanh_Kernel(float *dst, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = tanhf(dst[tid]);
	}
}
__global__
void ReLU_Kernel(float *dst, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m)
		dst[tid] = fmaxf(0.f, dst[tid]);
}
__global__
void LReLU_Kernel(float *dst, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m)
		dst[tid] = fmaxf(dst[tid] * LRELU_LEAK, dst[tid]);
}
__global__
void Sine_Kernel(float *dst, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = col * m + row;
	if (col < k && row < m) {
		dst[tid] = __sinf(dst[tid]);
	}
}
void d_activate(d_Matrix *dst, const Activation act) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	switch (act) {
		case Sigmoid:
			Sigmoid_Kernel << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
			break;
		case Tanh:
			Tanh_Kernel << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
			break;
		case ReLU:
			ReLU_Kernel << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
			break;
		case LReLU:
			LReLU_Kernel << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
			break;
		case Sine:
			Sine_Kernel << < dimGrid(m, k), dimBlock() >> > (dst->d_data(), m, k);
			break;
		case Linear: //fall through
		default:
			break;
	}
}
__global__
void backSigmoid_Kernel(float *dst, const float *d_A, const uint m, const uint n, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		const uint index = col * m + row;
		const float x = d_A[index];
		dst[index] = __fmul_rd(x, __fsub_rd(1.f, x));
	}
} /* dst = (d_W.T * d_dZ) (*) d_A */
void d_backSigmoid(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint n = d_W->rows(); //reverse for transpose
	const uint k = d_dZ->cols();
	backSigmoid_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
	d_catchErr();
}
__global__
void backTanh_Kernel(float *dst, const float *d_A, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		const uint index = col * m + row;
		const float x = d_A[index];
		dst[index] = __fmul_rd(dst[index], __fsub_rd(1.f, __fmul_rd(x, x)));
	}
} /* dst = (d_W.T * d_dZ) (*) 1 - d_A^2 */
void d_backTanh(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint k = d_dZ->cols();
	backTanh_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, k);
	d_catchErr();
}
__global__
void backReLU_Kernel(float *dst, const float *d_A, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		dst[row * k + col] *= (d_A[row * k + col] > 0.f ? 1.f : 0.f);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint k = d_dZ->cols();
	backReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, k);
	d_catchErr();
}
__global__
void backLReLU_Kernel(float *dst, const float *d_A, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		dst[row * k + col] *= (d_A[row * k + col] > 0.f ? 1.f : LRELU_LEAK);
	}
} /* dst = (d_W.T * d_dZ) (*) (d_A > 0 ? 1 : 0) */
void d_backLReLU(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint k = d_dZ->cols();
	backLReLU_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, k);
	d_catchErr();
}
__global__
void backSine_Kernel(float *dst, const float *d_A, const uint m, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		dst[row * k + col] *= cos(d_A[row * k + col]);
	}
} /* dst = cos(d_W.T * d_dZ) */
void d_backSine(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint k = d_dZ->cols();
	backSine_Kernel << <dimGrid(m, k), dimBlock() >> >
		(dst->d_data(), d_A->d_data(), m, k);
	d_catchErr();
}
__global__
void scaleValues_Kernel(float *values, const uint count, const float scale) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count) values[tid] *= scale;
}
/* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_A, const float coefficient) {
	d_mult_rhsT(dst, d_dZ, d_A);
	const uint count = uint(dst->size());
	if (count) scaleValues_Kernel<<<(count + 255) / 256, 256>>>(dst->d_data(), count, coefficient);
	d_catchErr();
}
__global__
void set_dW_Reg_Kernel(float *dst, const float *d_W, const float coefficient, const float regTerm, const uint count) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < count) {
		dst[tid] = coefficient * (dst[tid] + (regTerm * d_W[tid]));
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5f * learn * d_W) */
void d_set_dW_Reg(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_A, const d_Matrix *d_W, const float coefficient, const float regTerm) {
	d_mult_rhsT(dst, d_dZ, d_A);
	const uint count = uint(dst->size());
	if (count) set_dW_Reg_Kernel<<<(count + 255) / 256, 256>>>(dst->d_data(), d_W->d_data(), coefficient, regTerm, count);
	d_catchErr();
}
void d_sumRows(d_Matrix* dst, const d_Matrix* src) {
	const int k = src->cols();
	d_Matrix ones = d_Matrix(k, 1);d_catchErr();
	d_set_elem(&ones, 1.f); d_catchErr();
	d_mult(dst, src, &ones); d_catchErr();
	ones.free(); d_catchErr();
}
/* dst = coeff * (srcA.SumOfRows) */
__global__
void set_db_Kernel(float *dst, const float *src, const uint rows, const uint cols, const float coefficient) {
	__shared__ float partial[256];
	const uint row = blockIdx.x;
	float sum = 0.f;
	for (uint col = threadIdx.x; col < cols; col += blockDim.x) {
		sum += src[col * rows + row];
	}
	partial[threadIdx.x] = sum;
	__syncthreads();
	for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}
	if (threadIdx.x == 0) dst[row] = coefficient * partial[0];
}
void d_set_db(d_Matrix *dst, const d_Matrix *d_dZ, const float coefficient) {
	const uint rows = uint(d_dZ->rows());
	if (rows) set_db_Kernel<<<rows, 256>>>(dst->d_data(), d_dZ->d_data(), rows, uint(d_dZ->cols()), coefficient);
	d_catchErr();
}
#define BETA1 0.9f
#define BETA2 (1.f-FLT_EPSILON)
__global__
void updateParameterADAM_Kernel(float *dst, const uint N, const float *d_derivative, float *d_momentum, float *d_momentumSqr, const float learn) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		const float derivative = d_derivative[tid];
		const float momentum = BETA1 * d_momentum[tid] + (1.f - BETA1) * derivative;
		const float momentumSqr = BETA2 * d_momentumSqr[tid] + (1.f - BETA2) * derivative * derivative;
		dst[tid] -= learn * (momentum / (1.f - (BETA1 * BETA1)) /
			(sqrtf(momentumSqr / (1.f - (BETA2 * BETA2))) + FLT_EPSILON));
		d_momentum[tid] = momentum;
		d_momentumSqr[tid] = momentumSqr;
	}
}
void d_updateParameterADAM(d_Matrix* dst, const d_Matrix* d_derivative, const d_Matrix* d_momentum, const d_Matrix* d_momentumSqr, const float learnRate) {
	const uint count = uint(dst->size());
	if (count) updateParameterADAM_Kernel<<<(count + 255) / 256, 256>>>(
		dst->d_data(), count, d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate);
	d_catchErr();
}
__global__
void updateParameter_Kernel(float *dst, const uint N, const float *d_derivative, const float learn) {
	const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < N) {
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, const d_Matrix* d_derivative, const float learnRate) {
	const uint count = uint(dst->size());
	if (count) updateParameter_Kernel<<<(count + 255) / 256, 256>>>(
		dst->d_data(), count, d_derivative->d_data(), learnRate);
	d_catchErr();
}
void d_square(d_Matrix* dst, const d_Matrix* src) {
	d_launch2D_elem(pfMult, dst, src->d_data(), src->d_data());
}
__global__
void finalCost_Kernel(float *dst, const float *weightSum, float *epochCost, const float regMult, const float coefficient, const float trainCount) {
	dst[0] = (dst[0] * coefficient) + (0.5f * (regMult * (weightSum[0] / (trainCount * 2.0f))));
	if (epochCost) atomicAdd(epochCost, dst[0]);
}
void d_calcCost(float *dst, float *weightSum, float *epochCost, const d_Matrix* d_err,
	const vector<d_Matrix>* d_modelWeights, const float regMult, const float coeff, const float trainLabelCount) {
	d_check(cudaMemsetAsync(dst, 0, sizeof(float)));
	d_check(cudaMemsetAsync(weightSum, 0, sizeof(float)));
	const uint errorCount = uint(d_err->size());
	if (errorCount) {
		const uint blocks = min((errorCount + 255) / 256, 1024u);
		sumSquares_Kernel<<<blocks, 256>>>(dst, d_err->d_data(), errorCount);
	}
	for (size_t i = 0; i + 1 < d_modelWeights->size(); ++i) {
		const d_Matrix& weights = d_modelWeights->at(i);
		const uint count = uint(weights.size());
		if (count) {
			const uint blocks = min((count + 255) / 256, 1024u);
			sumSquares_Kernel<<<blocks, 256>>>(weightSum, weights.d_data(), count);
		}
	}
	finalCost_Kernel<<<1, 1>>>(dst, weightSum, epochCost, regMult, coeff, trainLabelCount);
	d_catchErr();
}
__global__
void averageCost_Kernel(float *epochCost, const int batchCount) {
	epochCost[0] /= float(batchCount);
}
void d_averageCost(float *epochCost, const int batchCount) {
	averageCost_Kernel<<<1, 1>>>(epochCost, batchCount);
	d_catchErr();
}
void d_gatherColumns(float *dst, const float *src, const int *indices, const int rows, const int count) {
	const uint elementCount = uint(rows) * uint(count);
	if (elementCount) {
		gatherColumns_Kernel<<<(elementCount + 255) / 256, 256>>>(
			dst, src, indices, uint(rows), uint(count));
		d_catchErr();
	}
}