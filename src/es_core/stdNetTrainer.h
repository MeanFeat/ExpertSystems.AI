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

class NetTrainer {
public:
	NetTrainer();
	NetTrainer(Net *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm);
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
	float coeff;
	Eigen::MatrixXf trainData;
	Eigen::MatrixXf trainLabels;
	NetCache cache;
	NetParameters dropParams;
	NetTrainParameters trainParams;
	NetTrainParameters momentum;
	NetTrainParameters momentumSqr;
};
