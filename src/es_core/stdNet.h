#pragma once
#include "types.h"

/**
 * Activation Functions
 * Inline implementations for forward pass through various activation functions
 */

inline Eigen::MatrixXf CalcSigmoid(const Eigen::MatrixXf &in) {
	// Use Eigen's efficient array operations instead of manual loop
	return ((-1.f * in).array().exp() + 1.f).cwiseInverse();
}

inline Eigen::MatrixXf CalcTanh(const Eigen::MatrixXf &in) {
	return (in.array().tanh());
}

inline Eigen::MatrixXf CalcReLU(const Eigen::MatrixXf &in) {
	return (in.cwiseMax(0.f));
}

inline Eigen::MatrixXf CalcLReLU(const Eigen::MatrixXf &in) {
	constexpr float LEAKY_RELU_ALPHA = 0.01f;
	return in.unaryExpr([](float elem) { return elem > 0.0f ? elem : elem * LEAKY_RELU_ALPHA; });
}

inline Eigen::MatrixXf CalcSine(const Eigen::MatrixXf &in) {
	return (in.array().sin());
}

inline Eigen::MatrixXf Log(const Eigen::MatrixXf &in) {
	return in.array().log();
}

/**
 * Neural network parameters structure
 * Contains all weights, biases, and layer configurations
 */
struct NetParameters {
	std::vector<int> layerSizes;					// Number of neurons in each layer
	std::vector<Activation> layerActivations;		// Activation function for each layer
	std::vector<Eigen::MatrixXf> W;					// Weight matrices for each layer
	std::vector<Eigen::MatrixXf> b;					// Bias vectors for each layer
};

/**
 * Feed-forward neural network class
 * Supports multiple hidden layers with configurable activation functions
 */
class Net {
public:
	Net();
	Net(int inputSize, const std::vector<int> &hiddenSizes, int outputSize, const std::vector<Activation> &activations);
	Net(const std::string &fName);
	~Net();
	NetParameters &GetParams();
	const NetParameters &GetParams() const;
	void SetParams(std::vector<Eigen::MatrixXf> W, std::vector<Eigen::MatrixXf> b);
	static Eigen::MatrixXf Activate(const Eigen::MatrixXf &In, Activation act);
	Eigen::MatrixXf ForwardPropagation(const Eigen::MatrixXf &X) const;
	int GetDepth() const;
	void RandomInit(float scale);
	float GetSumOfWeights() const;
	int GetNeuronCount() const;
	std::string ToString() const;
	void SaveNetwork(const std::string &fName) const;
	void LoadNetwork(const std::string &fName);
	int GetInputSize() const;
	int GetOutputSize() const;
	int GetNodeCount() const;
private:
	void AddLayer(int a, int b);
	NetParameters params;
};
