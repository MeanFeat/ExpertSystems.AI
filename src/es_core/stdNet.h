#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/dense>
#include "types.h"

inline Eigen::MatrixXf CalcSigmoid(const Eigen::MatrixXf &in) {
	//return in.unaryExpr([](float elem) { return 1.f / (1.f + exp(-elem)); });
	//return ((-1.f*in).array().exp() + 1).cwiseInverse();
	Eigen::MatrixXf out = Eigen::MatrixXf(in.rows(), in.cols());
	for (int i = 0; i < in.size(); i++) {
		*(out.data() + i) = 1.f / (1.f + expf(-(*(in.data() + i))));
	}
	return out;
}

inline Eigen::MatrixXf CalcTanh(const Eigen::MatrixXf &in) {
	return (in.array().tanh());
}

inline Eigen::MatrixXf CalcReLU(const Eigen::MatrixXf &in) {
	return (in.cwiseMax(0.f));
}

inline Eigen::MatrixXf CalcLReLU(const Eigen::MatrixXf &in) {
	return in.unaryExpr([](float elem) { return elem > 0.0f ? elem : elem * 0.01f; });
}

inline Eigen::MatrixXf CalcSine(const Eigen::MatrixXf &in) {
	return (in.array().sin());
}

inline Eigen::MatrixXf Log(const Eigen::MatrixXf &in) {
	return in.array().log();
}

struct NetParameters {
	std::vector<int> layerSizes;
	std::vector<Activation> layerActivations;
	std::vector<Eigen::MatrixXf> W;
	std::vector<Eigen::MatrixXf> b;
};

class Net {
public:
	Net();
	Net(int inputSize, std::vector<int> hiddenSizes, int outputSize, std::vector<Activation> activations);
	Net(const std::string fName);
	~Net();
	NetParameters &GetParams();
	void SetParams(std::vector<Eigen::MatrixXf> W, std::vector<Eigen::MatrixXf> b);
	void AddLayer(int A, int B);
	static Eigen::MatrixXf Activate(const Eigen::MatrixXf &In, Activation act);
	Eigen::MatrixXf ForwardPropagation(const Eigen::MatrixXf &X);

	int Depth() {
		return (int)GetParams().layerSizes.size() - 1;
	}

	void SaveNetwork();
	void LoadNetwork(const std::string fName);

protected:
	NetParameters params;
};
