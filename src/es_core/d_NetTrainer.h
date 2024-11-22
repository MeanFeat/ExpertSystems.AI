#pragma once

#include "stdNet.h"
#include "d_Matrix.h"
#include "d_math.h"
using namespace Eigen;
struct d_NetTrainParameters {
	void clear()
	{
		d_W.clear();
		d_b.clear();
	}
	float coefficient;
	float learnCoeff;
	float learnMult;
	float learnRate;
	float regMod;
	float regMult;
	float regTerm;
	std::vector<d_Matrix> d_W;
	std::vector<d_Matrix> d_b;
	unsigned int trainExamplesCount;
};
struct d_NetTrainDerivatives {
	void clear()
	{
		d_dW.clear();
		d_db.clear();
	}
	std::vector<d_Matrix> d_dW;
	std::vector<d_Matrix> d_db;
};
struct d_NetCache {
	void clear()
	{
		d_A.clear();
		d_AT.clear();
		d_dZ.clear();
	}
	std::vector<d_Matrix> d_A;
	std::vector<d_Matrix> d_AT;
	std::vector<d_Matrix> d_dZ;
	float cost;
	float *d_cost;
};
struct d_NetProfiler {
	float backpropTime;
	float calcCostTime;
	float forwardTime;
	float updateTime;
	float visualizationTime;
};
class d_NetTrainer {
public:
	d_NetTrainer();
	void free();
	d_NetTrainer(Net *net, const Eigen::MatrixXf &data, const Eigen::MatrixXf &labels, float weightScale, float learnRate, float regTerm);
	~d_NetTrainer();
	//TODO: Make copy constructor
	static d_Matrix to_device(MatrixXf matrix);
	static MatrixXf to_host(d_Matrix d_matrix);
	d_NetTrainParameters &GetTrainParams();
	d_NetCache &GetCache();
	const d_NetProfiler *GetProfiler() const;
	Net *network;
	d_Matrix d_trainLabels;
	void RefreshHostNetwork() const;
	void TrainSingleEpoch();
	d_Matrix Forward(const d_Matrix &Input) const;
	float CalcCost(const d_Matrix& Test, const d_Matrix& Labels) const;
	float GetCost()
	{
		return GetCache().cost;
	}
	float GetCoeff() const {
		return trainParams.coefficient;
	}
	float GetRegMultiplier() const {
		return trainParams.regMult;
	}
	inline void ModifyLearningRate(const float m) {
		trainParams.learnMult = max(FLT_EPSILON, trainParams.learnMult + (m * trainParams.learnCoeff));
	}
	inline void SetLearningRate(const float rate)
	{
		trainParams.learnRate = rate;
	}
	inline void ModifyRegTerm(const float m) {
		trainParams.regMod = max(FLT_EPSILON, trainParams.regMod + (m * trainParams.learnCoeff));
	}
	inline void SetRegTerm(const float term)
    {
        trainParams.regTerm = term;
    }
	unsigned int GetTrainExamplesCount() const {
		return trainParams.trainExamplesCount;
	}
	d_NetTrainDerivatives GetDerivatives() {
		return derivative;
	}
	void ForwardTrain();
	void BackwardPropagation();
private:
	void CalcCost();
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetTrainDerivatives derivative;
	d_NetTrainDerivatives momentum;
	d_NetTrainDerivatives momentumSqr;
	d_NetProfiler profiler;
	void AddLayer(int A, int B);
	void UpdateParameters();
	void UpdateParametersADAM();
};