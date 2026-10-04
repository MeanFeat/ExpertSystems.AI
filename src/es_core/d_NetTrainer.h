#pragma once

#include "stdNet.h"
#include "d_Matrix.h"
#include "d_math.h"
using namespace Eigen;
using namespace std;
struct d_NetBaseStructure {
	static void flushMat(std::vector<d_Matrix> &matrices)
	{
		for (d_Matrix &matrix : matrices)
		{
			matrix.free();
		}
		matrices.clear();
	}
};
struct d_NetTrainParameters : public d_NetBaseStructure {
    void clear()
	{
		flushMat(d_W);
		flushMat(d_b);
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
struct d_NetTrainDerivatives : public d_NetBaseStructure{
	void clear()
	{
		flushMat(d_dW);
		flushMat(d_db);
	}
	std::vector<d_Matrix> d_dW;
	std::vector<d_Matrix> d_db;
};
struct d_NetCache  : public d_NetBaseStructure {
	void clear()
	{
		flushMat(d_A);
        flushMat(d_dZ);
	}
	std::vector<d_Matrix> d_A;
	std::vector<d_Matrix> d_dZ;
	float cost;
	float *d_cost;
};
enum d_NetBatchShuffleType {
	None,
	ShuffleRandom,
	SlideWindow
};
struct d_NetBatchTrainingData {
	void clear() const
	{
		d_Data.free();
		d_Labels.free();
	}
	d_NetBatchTrainingData() = default;
	d_NetBatchTrainingData(const MatrixXf &data, const MatrixXf &labels);
	d_Matrix d_Data;
	d_Matrix d_Labels;
};
struct d_NetBatchParams {
	d_NetBatchParams() : slideOffset(0), batchCount(1), shuffleType(None) {};
	d_NetBatchParams(const int inBatchCount, const d_NetBatchShuffleType inShuffleType = None) : slideOffset(0), shuffleType(inShuffleType)	{
		batchCount = max(1, inBatchCount);
	}
	~d_NetBatchParams() {
		batchDataPool.clear();
	};
	int GetBatchSize() const {
		assert(batchCount > 0);
		return int(GetTotalTrainingExamples() / batchCount);
	}
	int GetBatchCount() const {
		return batchCount;
	}
	int GetTotalTrainingExamples() const {
		return batchDataPool.d_Labels.cols();
	}
	void CreateBatchData(const MatrixXf &data, const MatrixXf &labels);
	void ShuffleData();
	void LoadBatchData(const int batchIndex, d_Matrix& Input, d_Matrix& Output);
	d_NetBatchShuffleType GetShuffleType() const {
		return shuffleType;
	}
	int slideOffset;
private:
	int batchCount;
	d_NetBatchShuffleType shuffleType; 
	d_NetBatchTrainingData batchDataPool;
	std::vector<int> shuffledBatchIndices;
};
struct d_NetProfiler {
	float backpropTime;
	float calcCostTime;
	float loadBatchData;
	float forwardTime;
	float updateTime;
	float visualizationTime;
};
class d_NetTrainer {
public:
	d_NetTrainer();
	d_NetTrainer(Net *net, const MatrixXf &data, const MatrixXf &labels, float weightScale, float learnRate, float regTerm, const d_NetBatchParams& batchParameters = d_NetBatchParams());
	~d_NetTrainer();
	void free();
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
	float GetCost()	{
		const float cost = GetCache().cost;
		check(isnan(cost) == false);
		return cost;
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
	unsigned int GetTotalTrainingExamples() const {
		return trainParams.trainExamplesCount;
	}
	d_NetTrainDerivatives GetDerivatives() {
		return derivative;
	}
	void ForwardTrain();
	void BackwardPropagation();
private:
	void CalcCost() const;
	d_NetCache cache;
	d_NetTrainParameters trainParams;
	d_NetBatchParams batchParams;
	d_NetTrainDerivatives derivative;
	d_NetTrainDerivatives momentum;
	d_NetTrainDerivatives momentumSqr;
	d_NetProfiler profiler;
	void AddLayer(int A, int B);
	void UpdateParameters();
	void UpdateParametersADAM();
};