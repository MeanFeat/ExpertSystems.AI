#include "es_core_pch.h"
#include "stdNetTrainer.h"

using namespace Eigen;
using namespace std;

//ADAM optimizer hyperparameters
static constexpr float b1 = 0.9f;			// First moment decay rate
static constexpr float b2 = 1.f - FLT_EPSILON;	// Second moment decay rate
static constexpr float b1_sqr = b1 * b1;
static constexpr float b2_sqr = b2 * b2;
static constexpr float inv_b1 = 1.f - b1;
static constexpr float inv_b2 = 1.f - b2;
static constexpr float inv_b_sq1 = 1.f - b1_sqr;
static constexpr float inv_b_sq2 = 1.f - b2_sqr;

// Leaky ReLU alpha parameter
static constexpr float LEAKY_RELU_ALPHA = 0.01f;

NetTrainer::NetTrainer()
	: network(nullptr)
	, coeff(0)
	, cache()
	, trainParams()
	, momentum()
	, momentumSqr() {
}

NetTrainer::NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, float weightScale, float learnRate, float regTerm) {
	assert(net->GetNodeCount());
	assert(data.size());
	assert(labels.size());
	network = net;
	trainData = data;
	trainLabels = labels;
	coeff = float(1.f / float(trainLabels.cols()));
	if (network->GetSumOfWeights() == 0.f) {
		network->RandomInit(weightScale);
	}
	trainParams.learningMod = 1.f / float(network->GetNeuronCount());
	trainParams.learningRate = learnRate;
	trainParams.regTerm = regTerm;
	for (int i = 1; i < int(network->GetParams().layerSizes.size()); ++i) {
		AddLayer(network->GetParams().layerSizes[i], network->GetParams().layerSizes[i - 1]);
	}
}

NetTrainer::~NetTrainer() = default;

NetTrainParameters &NetTrainer::GetTrainParams() {
	return trainParams;
}

NetCache &NetTrainer::GetCache() {
	return cache;
}
void NetTrainer::AddLayer(const int A, const int B) {
	cache.Z.emplace_back(MatrixXf::Zero(0, 0));
	cache.A.emplace_back(MatrixXf::Zero(0, 0));
	trainParams.dW.emplace_back(MatrixXf::Zero(0, 0));
	trainParams.db.emplace_back(MatrixXf::Zero(0, 0));
	momentum.dW.emplace_back(MatrixXf::Zero(A, B));
	momentum.db.emplace_back(MatrixXf::Zero(A, 1));
	momentumSqr.dW.emplace_back(MatrixXf::Zero(A, B));
	momentumSqr.db.emplace_back(MatrixXf::Zero(A, 1));
}

MatrixXf NetTrainer::ForwardTrain() {
	MatrixXf lastOutput = MatrixXf(trainData.rows(), trainData.cols());
	lastOutput.noalias() = trainData;
	for (int i = 0; i < int(network->GetParams().layerSizes.size()) - 1; ++i) {
		MatrixXf weighed = network->GetParams().W[i] * lastOutput;
		cache.Z[i].noalias() = (weighed).colwise() + VectorXf(network->GetParams().b[i]);
		lastOutput = Net::Activate(cache.Z[i], network->GetParams().layerActivations[i]);
		cache.A[i].noalias() = lastOutput;
	}
	return lastOutput;
}

float NetTrainer::CalcCost(const MatrixXf &h, const MatrixXf &Y) const {
	float sumSqrW = 0.f;
	for (int w = 0; w < int(network->GetParams().W.size()) - 1; ++w) {
		sumSqrW += (network->GetParams().W[w].array() * network->GetParams().W[w].array()).sum();
	}
	const float regCost = 0.5f * float((trainParams.regTerm*trainParams.learningMod) * (sumSqrW / (2.f * float(trainLabels.cols()))));
	MatrixXf diff = Y - h;
	const float LocalCoeff = 1.f / float(trainLabels.cols());
	return ((diff.array() * diff.array()).sum() * LocalCoeff) + regCost;
}

void NetTrainer::ModifyLearningRate(const float m) {
	trainParams.learningRate = max(0.001f, trainParams.learningRate + m);
}

void NetTrainer::ModifyRegTerm(const float m) {
	trainParams.regTerm = max(FLT_EPSILON, trainParams.regTerm + m);
}

Eigen::MatrixXf NetTrainer::BackSigmoid(const Eigen::MatrixXf &wZ, const int index) const {
	return (wZ).cwiseProduct(MatrixXf::Ones(cache.A[index].rows(), cache.A[index].cols()) - cache.A[index]);
}

Eigen::MatrixXf NetTrainer::BackTanh(const Eigen::MatrixXf &wZ, const int index) {
	const MatrixXf invASqr = 1.f - cache.A[index].array().square();
	return (wZ).cwiseProduct(invASqr);
}

Eigen::MatrixXf NetTrainer::BackReLu(const Eigen::MatrixXf &wZ, const int index) const {
	return (wZ).cwiseProduct(cache.A[index].unaryExpr([](const float elem) { return elem > 0.f ? 1.f : 0.f; }));
}

Eigen::MatrixXf NetTrainer::BackLReLu(const Eigen::MatrixXf &wZ, const int index) const {
	return (wZ).cwiseProduct(cache.A[index].unaryExpr([](float elem) { return elem > 0.f ? 1.f : LEAKY_RELU_ALPHA; }));
}

Eigen::MatrixXf NetTrainer::BackSine(const Eigen::MatrixXf &wZ, int index) {
	return (wZ).cwiseProduct(MatrixXf(cache.A[index].array().cos()));
}

MatrixXf NetTrainer::BackActivation(const MatrixXf &dZ, const int layerIndex) {
	const MatrixXf wTdz = network->GetParams().W[layerIndex + 1].transpose() * dZ;
	switch (network->GetParams().layerActivations[layerIndex]) {
	case Sigmoid:
		return BackSigmoid(wTdz, layerIndex);
	case Tanh:
		return BackTanh(wTdz, layerIndex);
	case ReLU:
		return BackReLu(wTdz, layerIndex);
	case LReLU:
		return BackLReLu(wTdz, layerIndex);
	case Sine:
		return BackSine(wTdz, layerIndex);
	case Linear:
	default:
		return wTdz;
	}
}
/**
 * Compute gradients for a single layer during backpropagation
 * @param dZ Error gradient from upper layer
 * @param lowerA Activations from lower layer
 * @param layerIndex Index of current layer
 */
void NetTrainer::BackLayer(MatrixXf &dZ, const MatrixXf &lowerA, const int layerIndex) {
	dZ = BackActivation(dZ, layerIndex);
	const float lambda = 0.5f * (trainParams.regTerm * trainParams.learningMod);
	trainParams.dW[layerIndex] = coeff * MatrixXf((dZ * lowerA.transpose()).array() + (lambda * network->GetParams().W[layerIndex]).array());
	trainParams.db[layerIndex] = dZ.rowwise().sum();
	trainParams.db[layerIndex] *= coeff;
}

/**
 * Perform backpropagation through all layers
 * Computes gradients for weights and biases using the chain rule
 */
void NetTrainer::BackwardPropagation() {
	// Output layer gradient
	MatrixXf dZ = MatrixXf(cache.A.back() - trainLabels);
	trainParams.dW.back() = coeff * (dZ * cache.A[cache.A.size() - 2].transpose());
	trainParams.db.back() = coeff * dZ.rowwise().sum();
	
	// Hidden layers gradients (from output to input)
	for (int l = int(network->GetParams().layerActivations.size()) - 2; l >= 1; --l) {
		BackLayer(dZ, cache.A[l - 1], l);
	}
	
	// Input layer gradient
	BackLayer(dZ, trainData, 0);
}
}
/**
 * Update network parameters using standard gradient descent
 */
void NetTrainer::UpdateParameters() const {
	const float learnRate = (trainParams.learningRate*trainParams.learningMod);
	for (int i = 0; i < (int)trainParams.dW.size(); ++i) {
		network->GetParams().W[i] -= learnRate * trainParams.dW[i];
		network->GetParams().b[i] -= learnRate * trainParams.db[i];
	}
}

/**
 * Update network parameters using Adam optimizer
 * Implements adaptive learning rates with momentum
 */
void NetTrainer::UpdateParametersAdam() {
	const float learnRate = (trainParams.learningRate*trainParams.learningMod);
	for (int i = 0; i < int(trainParams.dW.size()); ++i) {
		NetTrainParameters vCorrected = momentum;
		NetTrainParameters sCorrected = momentumSqr;
		momentum.dW[i] = b1 * momentum.dW[i] + (inv_b1)* trainParams.dW[i];
		momentum.db[i] = b1 * momentum.db[i] + (inv_b1)* trainParams.db[i];
		vCorrected.dW[i] = momentum.dW[i] / (inv_b_sq1);
		vCorrected.db[i] = momentum.db[i] / (inv_b_sq1);
		momentumSqr.dW[i] = (b2 * momentumSqr.dW[i]) + ((inv_b2)* MatrixXf(trainParams.dW[i].array() * trainParams.dW[i].array()));
		momentumSqr.db[i] = (b2 * momentumSqr.db[i]) + ((inv_b2)* MatrixXf(trainParams.db[i].array() * trainParams.db[i].array()));
		sCorrected.dW[i] = momentumSqr.dW[i] / (inv_b_sq2);
		sCorrected.db[i] = momentumSqr.db[i] / (inv_b_sq2);
		network->GetParams().W[i] -= learnRate * MatrixXf(vCorrected.dW[i].array() / (sCorrected.dW[i].array().sqrt() + FLT_EPSILON));
		network->GetParams().b[i] -= learnRate * MatrixXf(vCorrected.db[i].array() / (sCorrected.db[i].array().sqrt() + FLT_EPSILON));
	}
}
void NetTrainer::BuildDropoutMask() {
	dropParams = network->GetParams();
	for (int i = 0; i < int(network->GetParams().W.size()) - 1; ++i) {
		for (int row = 0; row < network->GetParams().W[i].rows(); ++row) {
			const float val = (float(rand()) / (RAND_MAX));  // NOLINT(concurrency-mt-unsafe)
			if (val > 1.f) {
				dropParams.W[i].row(row) = MatrixXf::Zero(1, network->GetParams().W[i].cols());
				dropParams.b[i].row(row) = MatrixXf::Zero(1, network->GetParams().b[i].cols());
			}
		}
	}
}
void NetTrainer::TrainSingleEpoch() {
	//BuildDropoutMask();
	cache.cost = CalcCost(ForwardTrain(), trainLabels);
	BackwardPropagation();
	UpdateParametersAdam();
}