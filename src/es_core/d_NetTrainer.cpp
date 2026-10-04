#include "es_core_pch.h"
#include "d_NetTrainer.h"
#include <numeric>
#include <random>
using namespace Eigen;
using namespace std;
static cudaStream_t cuda_stream_default = nullptr;
static cudaStream_t cuda_stream_load = nullptr;
static cudaStream_t cuda_stream_cublas = nullptr;
cudaEvent_t start, stop;
static void createStreams() {
	if (!cuda_stream_default) cudaStreamCreate(&cuda_stream_default);
	if (!cuda_stream_load) cudaStreamCreate(&cuda_stream_load);
	if (!cuda_stream_cublas) cudaStreamCreate(&cuda_stream_cublas);
}
// Streams are shared statics: reset them so later to_device/to_host calls don't use destroyed handles
static void destroyStreams() {
	cudaDeviceSynchronize();
	cublasSetStream(cublasHandle, nullptr);
	for (cudaStream_t* s : { &cuda_stream_default, &cuda_stream_load, &cuda_stream_cublas }) {
		if (*s) cudaStreamDestroy(*s);
		*s = nullptr;
	}
}
void d_NetBatchParams::CreateBatchData(const MatrixXf& data, const MatrixXf& labels) {
	shuffledBatchIndices.resize(data.cols());
	iota(shuffledBatchIndices.begin(), shuffledBatchIndices.end(), 0);
	batchDataPool = d_NetBatchTrainingData(data, labels);
}
void d_NetBatchParams::ShuffleData() {
	random_device rd;
	mt19937 g(rd());
	shuffle(shuffledBatchIndices.begin(), shuffledBatchIndices.end(), g);
}
void d_NetBatchParams::LoadBatchData(const int batchIndex, d_Matrix& Input, d_Matrix& Output) {
	const int start_idx = batchIndex * GetBatchSize();
	const int end_idx = start_idx + GetBatchSize();
	const int dataRows = Input.rows();
	const int labelRows = Output.rows();
	switch (shuffleType) {
		case SlideWindow: {
			int offset_start = start_idx + slideOffset;
			int offset_end = end_idx + slideOffset;
			if (offset_start >= GetTotalTrainingExamples())
			{
				offset_start -= GetTotalTrainingExamples();
				offset_end -= GetTotalTrainingExamples();
			}
			if (offset_end < GetTotalTrainingExamples()) {
				cublasSetMatrixAsync(dataRows, GetBatchSize(), sizeof(float), batchDataPool.d_Data.d_data() + offset_start * dataRows, dataRows, Input.d_data(), dataRows, cuda_stream_load);
				cublasSetMatrixAsync(labelRows, GetBatchSize(), sizeof(float), batchDataPool.d_Labels.d_data() + offset_start * labelRows, labelRows, Output.d_data(), labelRows, cuda_stream_load);
			}
			else {
				const int wrapEnd = offset_end - GetTotalTrainingExamples();
				const int validSize = GetTotalTrainingExamples() - offset_start;
				cublasSetMatrixAsync(dataRows, validSize, sizeof(float), batchDataPool.d_Data.d_data() + offset_start * dataRows, dataRows, Input.d_data(), dataRows, cuda_stream_load);
				cublasSetMatrixAsync(labelRows, validSize, sizeof(float), batchDataPool.d_Labels.d_data() + offset_start * labelRows, labelRows, Output.d_data(), labelRows, cuda_stream_load);
				cublasSetMatrixAsync(dataRows, wrapEnd, sizeof(float), batchDataPool.d_Data.d_data(), dataRows, Input.d_data() + validSize * dataRows, dataRows, cuda_stream_load);
				cublasSetMatrixAsync(labelRows, wrapEnd, sizeof(float), batchDataPool.d_Labels.d_data(), labelRows, Output.d_data() + validSize * labelRows, labelRows, cuda_stream_load);
			}
		}
		break;
		case ShuffleRandom:
			for (int i = start_idx; i < end_idx; ++i) {
				const int idx = shuffledBatchIndices[i];
				const int d_idx = (i - start_idx);
				d_check(cudaMemcpyAsync(Input.d_data() + d_idx * dataRows, batchDataPool.d_Data.d_data() + idx * dataRows, dataRows * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_load));
				d_check(cudaMemcpyAsync(Output.d_data() + d_idx * labelRows, batchDataPool.d_Labels.d_data() + idx * labelRows, labelRows * sizeof(float), cudaMemcpyHostToDevice, cuda_stream_load));
			}
			break;
		case None: // fallthrough
		default: {
			cublasSetMatrixAsync(dataRows, GetBatchSize(), sizeof(float), batchDataPool.d_Data.d_data() + start_idx * dataRows, dataRows, Input.d_data(), dataRows, cuda_stream_load);
			cublasSetMatrixAsync(labelRows, GetBatchSize(), sizeof(float), batchDataPool.d_Labels.d_data() + start_idx * labelRows, labelRows, Output.d_data(), labelRows, cuda_stream_load);
		}
	}
	cudaStreamSynchronize(cuda_stream_load);
	d_catchErr();
}
d_Matrix d_NetTrainer::to_device(MatrixXf matrix) {
	d_mathInit();
	const int rows = int(matrix.rows());
	const int cols = int(matrix.cols());
	d_Matrix d_matrix = d_Matrix(rows, cols);
	cublasSetMatrixAsync(rows, cols, sizeof(float), matrix.data(), rows, d_matrix.d_data(), rows, cuda_stream_load); d_catchErr();
	return d_matrix;
}
MatrixXf d_NetTrainer::to_host(d_Matrix d_matrix) {
	const int rows = d_matrix.rows();
	const int cols = d_matrix.cols();
	MatrixXf out = MatrixXf(rows, cols);
	cublasGetMatrixAsync(rows, cols, sizeof(float), d_matrix.d_data(), rows, out.data(), rows, cuda_stream_load); d_catchErr();
	return out;
}
d_NetBatchTrainingData::d_NetBatchTrainingData(const MatrixXf& data, const MatrixXf& labels) {
	d_Data.setShape(int(data.rows()), int(data.cols()));
	d_Labels.setShape(int(labels.rows()), int(labels.cols()));
	d_check(cudaMallocHost(reinterpret_cast<void**>(&d_Data.d_data()), d_Data.memSize()));
	d_check(cudaMallocHost(reinterpret_cast<void**>(&d_Labels.d_data()), d_Labels.memSize()));
	d_check(cudaMemcpyAsync(d_Data.d_data(), data.data(), d_Data.memSize(), cudaMemcpyHostToHost, cuda_stream_load));
	d_check(cudaMemcpyAsync(d_Labels.d_data(), labels.data(), d_Labels.memSize(), cudaMemcpyHostToHost, cuda_stream_load));
}
d_NetTrainer::d_NetTrainer(): network(nullptr), cache(), trainParams(), batchParams(), profiler() {
	d_mathInit();
	createStreams();
}

d_NetTrainer::d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, const float weightScale, const float learnRate, const float regTerm, const d_NetBatchParams& batchParameters)
	: network(net), batchParams(batchParameters) {
	assert(net->GetNodeCount());
	assert(data.size());
	assert(labels.size());
	assert(batchParams.GetBatchCount() > 0);
#if _PROFILE
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif
	d_mathInit();
	createStreams();
	cublasSetStream(cublasHandle, cuda_stream_cublas);
	cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH);
	trainParams.trainExamplesCount = uint(data.cols());
	
	batchParams.CreateBatchData(data, labels);
	cache.d_A.emplace_back(int(data.rows()), batchParams.GetBatchSize());
	d_trainLabels = d_Matrix(int(labels.rows()), batchParams.GetBatchSize());
	trainParams.regTerm = regTerm / float(batchParams.GetBatchCount());
	trainParams.coefficient = 1.f / float(batchParams.GetBatchSize());
	
	if (network->GetSumOfWeights() == 0.f) {
		network->RandomInit(weightScale);
	}
	for (int i = 0; i < network->GetDepth(); ++i) {
		MatrixXf& W = network->GetParams().W[i];
		MatrixXf& b = network->GetParams().b[i];
		trainParams.d_W.emplace_back(W.data(), int(W.rows()), int(W.cols()));
		trainParams.d_b.emplace_back(b.data(), int(b.rows()), int(b.cols()));
	}
	trainParams.learnRate = learnRate;
	trainParams.learnCoeff = 1.f / float(network->GetNodeCount());
	trainParams.learnMult = trainParams.learnRate*trainParams.learnCoeff;
	trainParams.regMod = trainParams.regTerm / float(network->GetNodeCount());
	trainParams.regMult = float(trainParams.regTerm * trainParams.learnCoeff);
	for (int h = 1; h < network->GetDepth() + 1; ++h) {
		AddLayer(network->GetParams().layerSizes[h], network->GetParams().layerSizes[h - 1]);
	}
	d_check(cudaMalloc(&cache.d_cost, sizeof(float)));
	d_check(cudaMallocHost(VOID_PTR(&cache.cost), sizeof(float)));
	batchParams.LoadBatchData(0, cache.d_A[0], d_trainLabels);
}
d_NetTrainer::~d_NetTrainer() {
	free();
	destroyStreams();
}
void d_NetTrainer::free() {
	trainParams.clear();
	cache.clear();
	derivative.clear();
	momentum.clear();
	momentumSqr.clear();
	d_check(cudaFree(cache.d_cost));
}
d_NetTrainParameters &d_NetTrainer::GetTrainParams() {
	return trainParams;
}
d_NetCache &d_NetTrainer::GetCache() {
	return cache;
}
void d_NetTrainer::RefreshHostNetwork() const {
	for (int i = 0; i < trainParams.d_W.size(); ++i) {
		network->GetParams().W[i] = to_host(trainParams.d_W[i]);
		network->GetParams().b[i] = to_host(trainParams.d_b[i]);
	}
}
const d_NetProfiler *d_NetTrainer::GetProfiler() const {
	return &profiler;
}
void d_NetTrainer::AddLayer(int A, int B) {
	const int Cols = batchParams.GetBatchCount() > 1 ? batchParams.GetBatchSize() : GetTotalTrainingExamples();
	cache.d_A.emplace_back(A, Cols);
	cache.d_dZ.emplace_back(A, Cols);
	derivative.d_dW.emplace_back(A, B);
	derivative.d_db.emplace_back(A, 1);
	momentum.d_dW.emplace_back(A, B);
	momentum.d_db.emplace_back(A, 1);
	momentumSqr.d_dW.emplace_back(A, B);
	momentumSqr.d_db.emplace_back(A, 1);
}
d_Matrix d_NetTrainer::Forward(const d_Matrix &Input) const {
	d_Matrix PreviousLayer = Input;
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_Matrix Layer(trainParams.d_W[i].rows(), PreviousLayer.cols() );
		d_forwardLayer(&Layer, &trainParams.d_W[i], &PreviousLayer, &trainParams.d_b[i]);
		d_activate(&Layer, network->GetParams().layerActivations[i]);
		PreviousLayer = Layer;
	}
	return PreviousLayer;
}
void d_NetTrainer::ForwardTrain() {
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_forwardLayer(&cache.d_A[i + 1], &trainParams.d_W[i], &cache.d_A[i], &trainParams.d_b[i]);
		d_activate(&cache.d_A[i + 1], network->GetParams().layerActivations[i]);
	}
}
float d_NetTrainer::CalcCost(const d_Matrix& Test, const d_Matrix& Labels) const {
	float *d_cost;
	float cost;
	d_check(cudaMalloc(&d_cost, sizeof(float)));
	d_Matrix Error = d_Matrix(Test.rows(), Test.cols());
	d_subtract_elem(&Error, Test, Labels);
	d_calcCost(d_cost, &Error, &trainParams.d_W, GetRegMultiplier(), 1.f/float(Labels.cols()), float(Labels.cols())); d_catchErr();
	d_check(cudaMemcpyAsync(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_default));
	d_check(cudaFree(d_cost));
	return cost;
}
void d_NetTrainer::CalcCost() const {
	d_calcCost(cache.d_cost, &cache.d_dZ.back(), &trainParams.d_W, GetRegMultiplier(), GetCoeff(), float(GetTotalTrainingExamples())); d_catchErr();
}
void d_NetTrainer::BackwardPropagation() {
	d_subtract_elem(&cache.d_dZ.back(), cache.d_A.back(), d_trainLabels); d_catchErr();
	d_Matrix d_ATLast(cache.d_A[cache.d_A.size() - 2].cols(), cache.d_A[cache.d_A.size() - 2].rows()); d_catchErr();
	d_transpose(&d_ATLast, &cache.d_A[cache.d_A.size() - 2]); d_catchErr();
	d_set_dW(&derivative.d_dW.back(), &cache.d_dZ.back(), &d_ATLast, GetCoeff()); d_catchErr();
	d_set_db(&derivative.d_db.back(), &cache.d_dZ.back(), GetCoeff()); d_catchErr();
	for (int l = int(network->GetParams().layerActivations.size() - 2); l >= 0; --l) {
		switch (network->GetParams().layerActivations[l]) {
		case Sigmoid:
			d_backSigmoid(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Tanh:
			d_backTanh(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case ReLU:
			d_backReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case LReLU:
			d_backLReLU(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Sine:
			d_backSine(&cache.d_dZ[l], &trainParams.d_W[l + 1], &cache.d_dZ[l + 1], &cache.d_A[l + 1]);
			break;
		case Linear:
		default:
			break;
		}
		d_Matrix d_AT = d_Matrix(cache.d_A[l].cols(), cache.d_A[l].rows());
		d_transpose(&d_AT, &cache.d_A[l]);
		d_set_dW_Reg(&derivative.d_dW[l], &cache.d_dZ[l], &d_AT, &trainParams.d_W[l], GetCoeff(), 0.5f * trainParams.regMod);
		d_set_db(&derivative.d_db[l], &cache.d_dZ[l], GetCoeff());
		d_AT.free();
	}
}
void d_NetTrainer::UpdateParameters() {
	for (int i = 0; i < int(derivative.d_dW.size()); ++i) {
		d_updateParameter(&trainParams.d_W[i], &derivative.d_dW[i], trainParams.learnMult);
		d_updateParameter(&trainParams.d_b[i], &derivative.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::UpdateParametersADAM() {
	for (int i = 0; i < int(derivative.d_dW.size()); ++i) {
		d_updateParameterADAM(&trainParams.d_W[i], &derivative.d_dW[i], &momentum.d_dW[i], &momentumSqr.d_dW[i], trainParams.learnMult);
		d_updateParameterADAM(&trainParams.d_b[i], &derivative.d_db[i], &momentum.d_db[i], &momentumSqr.d_db[i], trainParams.learnMult);
	}
}
void d_NetTrainer::TrainSingleEpoch() {
	float totalCost = 0.f;
	for (int i = 0; i < batchParams.GetBatchCount(); ++i) {
		float batchCost = 0.f;
		d_profile(start, stop, &profiler.loadBatchData, batchParams.LoadBatchData(i, cache.d_A[0], d_trainLabels));	d_catchErr();
		d_profile(start, stop, &profiler.forwardTime,	ForwardTrain());			d_catchErr();
		d_profile(start, stop, &profiler.backpropTime,	BackwardPropagation());		d_catchErr();
		d_profile(start, stop, &profiler.updateTime,	UpdateParametersADAM());	d_catchErr();
		d_profile(start, stop, &profiler.calcCostTime,	CalcCost());				d_catchErr();
		d_check(cudaMemcpyAsync(&batchCost, cache.d_cost, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_default));
		totalCost += batchCost;
	}
	if (batchParams.GetShuffleType() == SlideWindow) {
		const int randStep = 1 + rand() % batchParams.GetBatchSize();
		batchParams.slideOffset = (batchParams.slideOffset + randStep) % GetTotalTrainingExamples();
	}
	else if (batchParams.GetShuffleType() == ShuffleRandom) {
		batchParams.ShuffleData();
	}
	cache.cost = totalCost / float(batchParams.GetBatchCount());
}