#include "es_core_pch.h"
#include "d_NetTrainer.h"
#include <numeric>
#include <random>
using namespace Eigen;
using namespace std;
cudaEvent_t start, stop;
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
void d_NetBatchParams::LoadBatchData(const int batchIndex, d_Matrix& Input, d_Matrix& Output, int *deviceIndices) {
	const int start_idx = batchIndex * GetBatchSize();
	const int batchSize = GetBatchSize();
	const int dataRows = Input.rows();
	const int labelRows = Output.rows();
	const int examples = GetTotalTrainingExamples();
	switch (shuffleType) {
		case SlideWindow: {
			const int offsetStart = (start_idx + slideOffset) % examples;
			const int validSize = min(batchSize, examples - offsetStart);
			if (validSize == batchSize) {
				d_check(cudaMemcpyAsync(Input.d_data(), batchDataPool.d_Data.d_data() + offsetStart * dataRows,
					batchSize * dataRows * sizeof(float), cudaMemcpyDeviceToDevice));
				d_check(cudaMemcpyAsync(Output.d_data(), batchDataPool.d_Labels.d_data() + offsetStart * labelRows,
					batchSize * labelRows * sizeof(float), cudaMemcpyDeviceToDevice));
			}
			else {
				const int wrapEnd = batchSize - validSize;
				d_check(cudaMemcpyAsync(Input.d_data(), batchDataPool.d_Data.d_data() + offsetStart * dataRows,
					validSize * dataRows * sizeof(float), cudaMemcpyDeviceToDevice));
				d_check(cudaMemcpyAsync(Output.d_data(), batchDataPool.d_Labels.d_data() + offsetStart * labelRows,
					validSize * labelRows * sizeof(float), cudaMemcpyDeviceToDevice));
				d_check(cudaMemcpyAsync(Input.d_data() + validSize * dataRows, batchDataPool.d_Data.d_data(),
					wrapEnd * dataRows * sizeof(float), cudaMemcpyDeviceToDevice));
				d_check(cudaMemcpyAsync(Output.d_data() + validSize * labelRows, batchDataPool.d_Labels.d_data(),
					wrapEnd * labelRows * sizeof(float), cudaMemcpyDeviceToDevice));
			}
		}
		break;
		case ShuffleRandom:
			d_check(cudaMemcpyAsync(deviceIndices, shuffledBatchIndices.data() + start_idx,
				batchSize * sizeof(int), cudaMemcpyHostToDevice));
			d_gatherColumns(Input.d_data(), batchDataPool.d_Data.d_data(), deviceIndices, dataRows, batchSize);
			d_gatherColumns(Output.d_data(), batchDataPool.d_Labels.d_data(), deviceIndices, labelRows, batchSize);
			break;
		case None: // fallthrough
		default: {
			d_check(cudaMemcpyAsync(Input.d_data(), batchDataPool.d_Data.d_data() + start_idx * dataRows,
				batchSize * dataRows * sizeof(float), cudaMemcpyDeviceToDevice));
			d_check(cudaMemcpyAsync(Output.d_data(), batchDataPool.d_Labels.d_data() + start_idx * labelRows,
				batchSize * labelRows * sizeof(float), cudaMemcpyDeviceToDevice));
		}
	}
	d_catchErr();
}
d_Matrix d_NetTrainer::to_device(MatrixXf matrix) {
	d_mathInit();
	const int rows = int(matrix.rows());
	const int cols = int(matrix.cols());
	d_Matrix d_matrix(matrix.data(), rows, cols);
	return d_matrix;
}
MatrixXf d_NetTrainer::to_host(const d_Matrix& d_matrix) {
	const int rows = d_matrix.rows();
	const int cols = d_matrix.cols();
	MatrixXf out = MatrixXf(rows, cols);
	d_check(cudaMemcpy(out.data(), d_matrix.d_data(), d_matrix.memSize(), cudaMemcpyDeviceToHost));
	return out;
}
d_NetBatchTrainingData::d_NetBatchTrainingData(const MatrixXf& data, const MatrixXf& labels)
	: d_Data(data.data(), int(data.rows()), int(data.cols())),
	  d_Labels(labels.data(), int(labels.rows()), int(labels.cols())) {}
d_NetTrainer::d_NetTrainer(): network(nullptr), cache(), trainParams(), batchParams(), d_Buffer(nullptr), d_batchIndices(nullptr), profiler() {
	d_mathInit();
}

d_NetTrainer::d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, const float weightScale, const float learnRate, const float regTerm, const d_NetBatchParams& batchParameters)
	: network(net), batchParams(batchParameters) {
	assert(net->GetNodeCount());
	assert(data.size());
	assert(labels.size());
	assert(data.cols() == labels.cols());
	assert(batchParams.GetBatchCount() > 0);
	assert(batchParams.GetBatchCount() <= data.cols());
#if _PROFILE
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif
	d_mathInit();
	if (cublasSetStream(cublasHandle, nullptr) != CUBLAS_STATUS_SUCCESS ||
		cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) {
		OutputDebugStringA("Failed to configure cuBLAS for the default stream.\n");
		exit(EXIT_FAILURE);
	}
	trainParams.trainExamplesCount = uint(data.cols());
	
	batchParams.CreateBatchData(data, labels);
	d_check(cudaMalloc(VOID_PTR(&d_batchIndices), batchParams.GetBatchSize() * sizeof(int)));
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
	d_check(cudaMalloc(VOID_PTR(&cache.d_cost), sizeof(float)));
	d_check(cudaMalloc(VOID_PTR(&cache.d_weightSum), sizeof(float)));
	d_check(cudaMalloc(VOID_PTR(&cache.d_epochCost), sizeof(float)));
	d_check(cudaMallocHost(VOID_PTR(&cache.hostCost), sizeof(float)));
	batchParams.LoadBatchData(0, cache.d_A[0], d_trainLabels, d_batchIndices);
}
d_NetTrainer::~d_NetTrainer() {
	free();
}
void d_NetTrainer::free() {
	trainParams.clear();
	cache.clear();
	derivative.clear();
	momentum.clear();
	momentumSqr.clear();
	if (d_batchIndices) d_check(cudaFree(d_batchIndices));
	if (d_Buffer) d_check(cudaFree(d_Buffer));
	d_batchIndices = nullptr;
	d_Buffer = nullptr;
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
	d_check(cudaMemsetAsync(momentum.d_dW.back().d_data(), 0, momentum.d_dW.back().memSize()));
	d_check(cudaMemsetAsync(momentum.d_db.back().d_data(), 0, momentum.d_db.back().memSize()));
	d_check(cudaMemsetAsync(momentumSqr.d_dW.back().d_data(), 0, momentumSqr.d_dW.back().memSize()));
	d_check(cudaMemsetAsync(momentumSqr.d_db.back().d_data(), 0, momentumSqr.d_db.back().memSize()));
}
void d_NetTrainer::BuildVisualization(const MatrixXf &screen, int * buffer, const int m, const int k) {
	const int size = m*k;
	d_check(cudaMalloc(&d_Buffer, size * sizeof(int)));
	d_VisualA.push_back(to_device(screen));
	for (int i = 0; i < network->GetDepth(); ++i) {
		d_VisualA.emplace_back(trainParams.d_W[i].rows(), d_VisualA[i].cols());
	}
}
void d_NetTrainer::Visualization(int *buffer, const int m, const int k, const bool discrete) {
	d_profile(start, stop, &profiler.visualizationTime,
		for (int i = 0; i < network->GetDepth(); ++i) {
			d_forwardLayer(&d_VisualA[i + 1], &trainParams.d_W[i], &d_VisualA[i], &trainParams.d_b[i]);
			d_activate(&d_VisualA[i + 1], network->GetParams().layerActivations[i]);
		}
	d_drawPixels(d_Buffer, m, k, d_VisualA.back().d_data(), discrete);
	d_check(cudaMemcpyAsync(buffer, d_Buffer, m*k * sizeof(int), cudaMemcpyDeviceToHost));
	); //d_profile
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
	float *d_weightSum;
	float cost;
	d_check(cudaMalloc(VOID_PTR(&d_cost), sizeof(float)));
	d_check(cudaMalloc(VOID_PTR(&d_weightSum), sizeof(float)));
	d_Matrix Error(Test);
	d_subtract_elem(&Error, Test, Labels);
	d_calcCost(d_cost, d_weightSum, nullptr, &Error, &trainParams.d_W, GetRegMultiplier(), 1.f / float(Labels.cols()), float(Test.cols())); d_catchErr();
	d_check(cudaMemcpy(&cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
	d_check(cudaFree(d_cost));
	d_check(cudaFree(d_weightSum));
	return cost;
}
void d_NetTrainer::CalcCost() const {
	d_calcCost(cache.d_cost, cache.d_weightSum, cache.d_epochCost, &cache.d_dZ.back(), &trainParams.d_W,
		GetRegMultiplier(), GetCoeff(), float(GetTotalTrainingExamples())); d_catchErr();
}
void d_NetTrainer::BackwardPropagation() {
	d_subtract_elem(&cache.d_dZ.back(), cache.d_A.back(), d_trainLabels); d_catchErr();
	d_set_dW(&derivative.d_dW.back(), &cache.d_dZ.back(), &cache.d_A[cache.d_A.size() - 2], GetCoeff()); d_catchErr();
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
		d_set_dW_Reg(&derivative.d_dW[l], &cache.d_dZ[l], &cache.d_A[l], &trainParams.d_W[l], GetCoeff(), 0.5f * trainParams.regMod);
		d_set_db(&derivative.d_db[l], &cache.d_dZ[l], GetCoeff());
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
	d_check(cudaMemsetAsync(cache.d_epochCost, 0, sizeof(float)));
	for (int i = 0; i < batchParams.GetBatchCount(); ++i) {
		d_profile(start, stop, &profiler.loadBatchData, batchParams.LoadBatchData(i, cache.d_A[0], d_trainLabels, d_batchIndices));	d_catchErr();
		d_profile(start, stop, &profiler.forwardTime,	ForwardTrain());			d_catchErr();
		d_profile(start, stop, &profiler.backpropTime,	BackwardPropagation());		d_catchErr();
		d_profile(start, stop, &profiler.updateTime,	UpdateParametersADAM());	d_catchErr();
		d_profile(start, stop, &profiler.calcCostTime,	CalcCost());				d_catchErr();
	}
	d_averageCost(cache.d_epochCost, batchParams.GetBatchCount());
	d_check(cudaMemcpyAsync(cache.hostCost, cache.d_epochCost, sizeof(float), cudaMemcpyDeviceToHost));
	d_check(cudaStreamSynchronize(nullptr));
	cache.cost = *cache.hostCost;
	if (batchParams.GetShuffleType() == SlideWindow) {
		const int randStep = 1 + rand() % batchParams.GetBatchSize();
		batchParams.slideOffset = (batchParams.slideOffset + randStep) % GetTotalTrainingExamples();
	}
	else if (batchParams.GetShuffleType() == ShuffleRandom) {
		batchParams.ShuffleData();
	}
}