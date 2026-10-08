#pragma once
#include "stdNet.h"

struct NetTrainParameters {
	std::vector<Eigen::MatrixXf> dW;
	std::vector<Eigen::MatrixXf> db;
	float learningRate;
	float learningMod;
	float regTerm;
};

struct NetCache {
	std::vector<Eigen::MatrixXf> Z;
	std::vector<Eigen::MatrixXf> A;
	float cost;
};

struct NetBatchParams {
	int batchCount = 1;
	NetBatchShuffleType shuffleType = None;
};

class NetTrainer {
public:
	NetTrainer();
	NetTrainer(Net *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm,
	           const NetBatchParams &batchParameters = NetBatchParams());
	~NetTrainer();

	Eigen::MatrixXf BackActivation(const Eigen::MatrixXf &dZ, int layerIndex);
	Eigen::MatrixXf BackLReLu(const Eigen::MatrixXf &wZ, int index) const;
	Eigen::MatrixXf BackReLu(const Eigen::MatrixXf &wZ, int index) const;
	Eigen::MatrixXf BackSigmoid(const Eigen::MatrixXf &wZ, int index) const;
	Eigen::MatrixXf BackSine(const Eigen::MatrixXf &wZ, int index);
	Eigen::MatrixXf BackTanh(const Eigen::MatrixXf &wZ, int index);
	Eigen::MatrixXf ForwardTrain();
	float CalcCost(const Eigen::MatrixXf &h, const Eigen::MatrixXf &Y) const;
	Net *network;
	NetCache &GetCache();
	NetTrainParameters &GetTrainParams();
	void AddLayer(const int A, const int B);
	void BackLayer(Eigen::MatrixXf &dZ, const Eigen::MatrixXf &lowerA, int layerIndex);
	void BackwardPropagation();
	void BuildDropoutMask();
	void ModifyLearningRate(float m);
	void ModifyRegTerm(float m);
	void UpdateParameters() const;
	void UpdateParametersAdam();
	void TrainSingleEpoch();

private:
	void LoadBatch(int batchIndex);
	void AdvanceBatchWindow();
	float coeff;
	// trainData/trainLabels hold the active batch; allData/allLabels hold the full set when batching.
	Eigen::MatrixXf trainData;
	Eigen::MatrixXf trainLabels;
	Eigen::MatrixXf allData;
	Eigen::MatrixXf allLabels;
	NetBatchParams batchParams;
	std::vector<int> shuffledIndices;
	int slideOffset = 0;
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};
