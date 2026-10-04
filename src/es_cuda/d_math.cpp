#include "d_math.h"

using namespace std;
__device__ ptrFunc d_pfAdd = __fadd_rn;
__device__ ptrFunc d_pfSub = __fsub_rn;
__device__ ptrFunc d_pfMult = __fmul_rn;
__device__ ptrFunc d_pfSet = _set_elem;
namespace {
	ptrFunc pfAdd;
	ptrFunc pfSub;
	ptrFunc pfMult;
	ptrFunc pfSet;
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
		cublasCreate(&cublasHandle); d_catchErr();
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
void backTanh_Kernel(float *dst, const float *d_A, const uint m, const uint n, const uint k) {
	extern __shared__ float shared[];
	float *dst_shared = shared;
	float *d_A_shared = dst_shared + blockDim.x * blockDim.y;
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		const uint index = col * m + row;
		const uint local_index = threadIdx.x * blockDim.y + threadIdx.y;
		dst_shared[local_index] = dst[index];
		d_A_shared[local_index] = d_A[index];
		__syncthreads();
		const float x = d_A_shared[local_index];
		dst_shared[local_index] = __fmul_rd(dst_shared[local_index], __fsub_rd(1.f, __fmul_rd(x, x)));
		dst[index] = dst_shared[local_index];
	}
} /* dst = (d_W.T * d_dZ) (*) 1 - d_A^2 */
void d_backTanh(d_Matrix *dst, const d_Matrix *d_W, const d_Matrix *d_dZ, const d_Matrix *d_A) {
	d_mult_lhsT(dst, d_W, d_dZ);
	const uint m = d_W->cols(); //reverse for transpose
	const uint n = d_W->rows(); //reverse for transpose
	const uint k = d_dZ->cols();
	auto sharedSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
	backTanh_Kernel << <dimGrid(m, k), dimBlock(), sharedSize >> >
		(dst->d_data(), d_A->d_data(), m, n, k);
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
} /* dst = coeff * (d_dZ * d_A.T) */
void d_set_dW(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_AT, const float coefficient) {
	d_mult(dst, d_dZ, d_AT); d_catchErr();
	d_mult_scalar(dst, coefficient); d_catchErr();
}
__global__
void set_dW_Reg_Kernel(float *dst, const float *d_W, const float regTerm, const uint m, const uint n, const uint k) {
	const uint row = GetRow();
	const uint col = GetCol();
	if (col < k && row < m) {
		dst[col * m + row] += (regTerm * d_W[col * m + row]);
	}
} /* dst = coeff * (d_dZ * d_A.T) (+) (0.5f * learn * d_W) */
void d_set_dW_Reg(d_Matrix* dst, const d_Matrix* d_dZ, const d_Matrix* d_AT, const d_Matrix *d_W, const float coefficient, const float regTerm) {
	const uint m = d_dZ->rows();
	const uint n = d_dZ->cols();
	const uint k = d_AT->cols();
	d_mult(dst, d_dZ, d_AT);
	set_dW_Reg_Kernel << <dimGrid(m, k), dimBlock() >> > (dst->d_data(), d_W->d_data(), regTerm, m, n, k);
	d_mult_scalar(dst, coefficient);
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
void d_set_db(d_Matrix* dst, const d_Matrix* d_dZ, const float coefficient) {
	d_sumRows(dst, d_dZ); d_catchErr();
	d_mult_scalar(dst, coefficient); d_catchErr();
}
#define BETA1 0.9f
#define BETA2 (1.f-FLT_EPSILON)
__global__
void updateParameterADAM_Kernel(float *dst, const uint N, const float *d_derivative, float *d_momentum, float *d_momentumSqr, const float learn, const uint k) {
	extern __shared__ float s_data[];
	float *s_derivative = s_data;
	const uint dim = (blockDim.x * blockDim.y);
	float *s_momentum = s_derivative + dim;
	float *s_momentumSqr = s_momentum + dim;

	const uint row = GetRow();
	const uint col = GetCol();
	const uint tid = row * k + col;
	
	const uint local_tid = threadIdx.y * blockDim.x + threadIdx.x;

	if (tid < N) {
		s_derivative[local_tid] = d_derivative[tid];
		s_momentum[local_tid] = d_momentum[tid];
		s_momentumSqr[local_tid] = d_momentumSqr[tid];

		__syncthreads(); 

		s_momentum[local_tid] = BETA1 * s_momentum[local_tid] + (1.f - BETA1) * s_derivative[local_tid];
		s_momentumSqr[local_tid] = BETA2 * s_momentumSqr[local_tid] + (1.f - BETA2) * (s_derivative[local_tid] * s_derivative[local_tid]) ;
		dst[tid] -= learn * (s_momentum[local_tid] / (1.f - (BETA1 * BETA1)) / (sqrtf(s_momentumSqr[local_tid] / (1.f - (BETA2 * BETA2))) + FLT_EPSILON));
	
		d_momentum[tid] = s_momentum[local_tid];
		d_momentumSqr[tid] = s_momentumSqr[local_tid];
	}
}
void d_updateParameterADAM(d_Matrix* dst, const d_Matrix* d_derivative, const d_Matrix* d_momentum, const d_Matrix* d_momentumSqr, const float learnRate) {
	const uint m = dst->rows();
	const uint k = dst->cols();
	const uint dim = dimBlock().x * dimBlock().y * 3;
	size_t sharedMemSize = dim * sizeof(float);
	updateParameterADAM_Kernel << <dimGrid(m, k), dimBlock(), sharedMemSize>> >(dst->d_data(), dst->size(), d_derivative->d_data(), d_momentum->d_data(), d_momentumSqr->d_data(), learnRate, k);
	d_catchErr();
}
__global__
void updateParameter_Kernel(float *dst, const uint N, const float *d_derivative, const float learn) {
	const uint tid = blockIdx.x;
	if (tid < N) {
		dst[tid] -= learn * d_derivative[tid];
	}
}
void d_updateParameter(d_Matrix* dst, const d_Matrix* d_derivative, const float learnRate) {
	updateParameter_Kernel << <dst->size(), 1 >> >
		(dst->d_data(), dst->size(), d_derivative->d_data(), learnRate);
	d_catchErr();
}
void d_square(d_Matrix* dst, const d_Matrix* src) {
	d_launch2D_elem(pfMult, dst, src->d_data(), src->d_data());
}
__global__
void finalCost_Kernel(float *dst, const float *sumTotal, const float regMult, const float trainCount) {
	dst[0] = dst[0] + (0.5f * (regMult * (sumTotal[0] / (trainCount*2.0f))));
}
void d_calcCost(float *dst, const d_Matrix* d_err, const vector<d_Matrix>* d_modelWeights, const float regMult, float const coeff, const float trainLabelCount) {
	d_Matrix *d_diff = new d_Matrix(d_err->rows(), d_err->cols());
	d_square(d_diff, d_err);
	d_sumMatrix(dst, d_diff);
	d_mult_scalar(dst, coeff, 1, 1);
	// Add Regularization
	d_Matrix d_sqrSumTotal = d_Matrix(1, 1);
	d_set_elem(d_sqrSumTotal.d_data(), 0.f);
	d_Matrix d_sqrSum = d_Matrix(1, 1);
	for (uint i = 0; i < uint(d_modelWeights->size()) - 1; ++i) {
		const d_Matrix *layerWeights = &d_modelWeights->at(i);
		d_Matrix d_squared(layerWeights->rows(), layerWeights->cols());
		d_square(&d_squared, layerWeights); d_catchErr();
		d_sumMatrix(d_sqrSum.d_data(), &d_squared); d_catchErr();
		d_launch_single_thread(pfAdd, d_sqrSumTotal.d_data(), d_sqrSum.d_data()); d_catchErr();
	}
	finalCost_Kernel << <1, 1 >> > (dst, d_sqrSumTotal.d_data(), regMult, trainLabelCount); d_catchErr();
	d_diff->free();
}