#include "es_core_pch.h"
#include "stdNet.h"
#include "es_parser.h"

using namespace Eigen;
using namespace std;
Net::Net() = default;
Net::Net(const int inputSize, const std::vector<int> &hiddenSizes, const int outputSize, const vector<Activation> &activations) {
	params.layerActivations = activations;
	params.layerSizes.push_back(inputSize);
	for (int l = 0; l < int(hiddenSizes.size()); ++l) {
		params.layerSizes.push_back(hiddenSizes[l]);
	}
	params.layerSizes.push_back(outputSize);
	AddLayer(hiddenSizes[0], inputSize);
	for (int h = 1; h < int(hiddenSizes.size()); ++h) {
		AddLayer(hiddenSizes[h], hiddenSizes[h - 1]);
	}
	AddLayer(outputSize, hiddenSizes.back());
}

Net::Net(const string &fName) {
	LoadNetworkJson(fName);
}

Net::~Net() {}
NetParameters &Net::GetParams() {
	return params;
}
const NetParameters &Net::GetParams() const {
	return params;
}
void Net::SetParams(const vector<MatrixXf>& W, const vector<MatrixXf>& b) {
	params.W = W;
	params.b = b;
}
void Net::AddLayer(const int a, const int b) {
	params.W.push_back(MatrixXf::Zero(a, b));
	params.b.push_back(MatrixXf::Zero(a, 1));
}
MatrixXf Net::Activate(const MatrixXf &In, const Activation act) {
	switch (act) {
	case Linear:
		return In;
	case Sigmoid:
		return CalcSigmoid(In);
	case Tanh:
		return CalcTanh(In);
	case ReLU:
		return CalcReLU(In);
	case LReLU:
		return CalcLReLU(In);
	case Sine:
		return CalcSine(In);
	default:
		return In;
	}
}

MatrixXf Net::ForwardPropagation(const MatrixXf &X) const
{
	MatrixXf lastOutput = X;
	for (int i = 0; i < (int)params.layerSizes.size() - 1; ++i) {
		MatrixXf weighed = params.W[i] * lastOutput;
		weighed.colwise() += (VectorXf)params.b[i];
		lastOutput = Net::Activate(weighed, params.layerActivations[i]);
	}
	return lastOutput;
}

int Net::GetDepth() const {
	return static_cast<int>(GetParams().layerSizes.size()) - 1;
}

void Net::RandomInit(const float scale) {
	for (int i = 0; i < GetDepth(); ++i) {
		MatrixXf *w = &GetParams().W[i];
		*w = MatrixXf::Random(w->rows(), w->cols()) * scale;
	}
}

float Net::GetSumOfWeights() {
	float result = 0.f;
	for (int i = 0; i < GetParams().W.size(); i++) {
		result += GetParams().W[i].sum();
	}
	return result;
}

int Net::GetNeuronCount() {
	int result = 0;
	for (int i = 0; i < GetDepth(); i++) {
		result += GetParams().layerSizes[i];
	}
	return result;
}

static Activation ReadActivation(const string& str) {
	if (strFind(str, "sigmoid")) {
		return Activation::Sigmoid;
	}
	else if (strFind(str, "tanh")) {
		return Activation::Tanh;
	}
	else if (strFind(str, "relu")) {
		return Activation::ReLU;
	}
	else if (strFind(str, "leaky_relu")) {
		return Activation::LReLU;
	}
	else {
		return Activation::Linear;
	}
}

static string WriteActivation(const Activation act) {
	switch (act) {
	case Sigmoid:	return "sigmoid";
	case Tanh:		return "tanh";
	case ReLU:		return "relu";
	case LReLU:		return "leaky_relu";
	case Sine:		return "sine";
	case Linear:	//fall through
	default: return "linear";
	}
}

static inline void OutputMatrix(std::ofstream &file, int tb, const std::string& shape, const std::string& vals, const Eigen::MatrixXf *m, const bool transposed = false) {
	OUT_LINE(file, tb, "\"" << shape << "\" :" << "[");
	if (transposed) {
		OUT_LINE(file, ++tb, m->cols() << ",");
		OUT_LINE(file, tb, m->rows());
	}
	else {
		OUT_LINE(file, ++tb, m->rows() << ",");
		OUT_LINE(file, tb, m->cols());
	}
	OUT_LINE(file, --tb, "],");

	OUT_LINE(file, tb, "\"" << vals << "\" : [");
	for (int c = 0; c < m->cols(); c++) {
		OUT_LINE(file, ++tb, (m->cols() > 1 ? "[" : ""));
		tb++;
		for (int r = 0; r < m->rows(); r++) {
			char ch[100];
			sprintf_s(ch, "%0.25f", *(m->data() + c * m->rows() + r));
			OUT_LINE(file, tb, ch << (r == m->rows() - 1 ? "" : ","));
		}
		OUT_LINE(file, --tb, (m->cols() > 1 && c < m->cols() - 1 ? "]," : "]"));
		--tb;
	}
}

void Net::SaveNetworkJson(const string& fName) {
	if (fName.empty())
		return;
	ofstream file(fName);
	int tb = 0;
	const NetParameters *p = &params;
	for (int lyrIdx = 0; lyrIdx < GetDepth(); lyrIdx++) {
		OUT_LINE(file, tb, "{");
		OUT_LINE(file, ++tb, "\"layer\" :" << lyrIdx << ",");
		OUT_LINE(file, tb, "\"activation\" :" << "\"" << WriteActivation(p->layerActivations[lyrIdx]) << "\",");
		OutputMatrix(file, tb, "weightShape", "weightVals", &p->W[lyrIdx], true);
		OUT_LINE(file, tb, "],");
		OutputMatrix(file, tb, "biasShape", "biasVals", &p->b[lyrIdx]);
		OUT_LINE(file, --tb, (lyrIdx < GetDepth() - 1 ? "}," : "}"));
	}
	file.close();
}

void Net::SaveNetwork(const string& fName) {
	if (fName.empty())
		return;
	ofstream file(fName, ios::binary);
	int depth = GetDepth();
	file.write((char*)&depth, sizeof(int));
	for (int lyrIdx = 0; lyrIdx < depth; lyrIdx++) {
		int rows = static_cast<int>(params.W[lyrIdx].rows());
		int cols = static_cast<int>(params.W[lyrIdx].cols());
		file.write((char*)&rows, sizeof(int));
		file.write((char*)&cols, sizeof(int));
		file.write((char*)params.W[lyrIdx].data(), rows * cols * sizeof(float));

		rows = static_cast<int>(params.b[lyrIdx].rows());
		cols = static_cast<int>(params.b[lyrIdx].cols());
		file.write((char*)&rows, sizeof(int));
		file.write((char*)&cols, sizeof(int));
		file.write((char*)params.b[lyrIdx].data(), rows * cols * sizeof(float));

		int act = static_cast<int>(params.layerActivations[lyrIdx]);
		file.write((char*)&act, sizeof(int));
	}
	file.close();
}

namespace
{
	enum class NetParseState {
		layer,
		activation,
		weightShape,
		biasShape,
		weightValues,
		biasValues,
		none
	};
}

void Net::LoadNetworkJson(const string& fName) {
	if (fName.empty())
		return;
	std::string line;
	ifstream file(fName);
	vector<int> shapeDims;
	MatrixXf tempMat;
	NetParseState state = NetParseState::none;
	int element = 0, layerIndex = 0;
	while (file.good()) {
		std::getline(file, line);
		std::stringstream iss(line);
		std::string val, lastVal;
		while (iss.good()) {
			std::getline(iss, val, ':');
			if (state == NetParseState::none) {
				if (strFind(lastVal, "layer")) {
					state = NetParseState::layer;
					strCast(&layerIndex, val);
				}
				else if (strFind(lastVal, "activation")) {
					state = NetParseState::activation;
					params.layerActivations.push_back(ReadActivation(val));
				}
				if (strFind(lastVal, "Shape")) {
					shapeDims.clear();
					state = strFind(lastVal, "weight") ? NetParseState::weightShape : NetParseState::biasShape;
				}
				else if (strFind(lastVal, "Vals")) {
					state = strFind(lastVal, "weight") ? NetParseState::weightValues : NetParseState::biasValues;
				}
				lastVal = val;
			}
			switch (state) {
			case NetParseState::none:
			case NetParseState::layer:
			case NetParseState::activation: //fall through
				state = NetParseState::none;
				break;
			case NetParseState::weightShape:
			case NetParseState::biasShape:	//fall through
			{
				if (!strFind(val, "]")) {
					std::getline(iss, val, ',');
					if (!strFind(val, "[")) {
						int temp;
						strCast(&temp, val);
						shapeDims.push_back(temp);
						assert(shapeDims.size() <= 2);
					}
				}
				else {
					if (state == NetParseState::weightShape) {
						if (layerIndex == 0) {
							params.layerSizes.push_back(shapeDims[0]);
						}
						params.layerSizes.push_back(shapeDims[1]);
					}
					tempMat = MatrixXf(shapeDims[0], shapeDims[1]);
					state = NetParseState::none;
				}
			}
			break;
			case NetParseState::weightValues:
			case NetParseState::biasValues://fall through
			{
				if (!strFind(val, "]")) {
					std::getline(iss, val, ',');
					if (!strFind(val, "[")) {
						float temp;
						strCast(&temp, val);
						if (shapeDims[0] != 1 && shapeDims[1] != 1) {
							int r = int(floor(element % shapeDims[1]));
							int c = int(floor(element / shapeDims[1]));
							*(tempMat.data() + ((r * shapeDims[0]) + c)) = temp;
							element++;
						}
						else {
							*(tempMat.data() + element++) = temp;
						}
					}
				}
				if (element >= tempMat.size()) {
					element = 0;
					if (state == NetParseState::weightValues) {
						params.W.push_back(tempMat.transpose());
					}
					else {
						params.b.push_back(tempMat);
					}
					state = NetParseState::none;
				}
			}
			break;
			default:
				state = NetParseState::none;
				break;
			}
		}
	}
	file.close();
}

void Net::LoadNetwork(const string& fName) {
	if (fName.empty())
		return;
	ifstream file(fName, ios::binary);
	int depth = 0;
	file.read((char*)&depth, sizeof(int));
	for (int lyrIdx = 0; lyrIdx < depth; lyrIdx++) {
		int rows = 0, cols = 0;
		file.read((char*)&rows, sizeof(int));
		file.read((char*)&cols, sizeof(int));
		MatrixXf w(rows, cols);
		file.read((char*)w.data(), rows * cols * sizeof(float));
		params.W.push_back(w);

		file.read((char*)&rows, sizeof(int));
		file.read((char*)&cols, sizeof(int));
		MatrixXf b(rows, cols);
		file.read((char*)b.data(), rows * cols * sizeof(float));
		params.b.push_back(b);

		int act = 0;
		file.read((char*)&act, sizeof(int));
		params.layerActivations.push_back(static_cast<Activation>(act));

		if (lyrIdx == 0) {
			params.layerSizes.push_back(static_cast<int>(w.rows()));
		}
		params.layerSizes.push_back(static_cast<int>(w.cols()));
	}
	file.close();
}

int Net::GetInputSize() const
{
	return params.layerSizes[0];
}
int Net::GetOutputSize() const
{
	return params.layerSizes.back();
}

int Net::GetNodeCount() {
	int count = 0;
	for (int i = 0; i < GetDepth(); i++) {
		count += GetParams().layerSizes[i];
	}
	return count;
}

string Net::ToString() {
	string str;
	const NetParameters *p = &params;
	for (int i = 0; i < GetDepth(); i++) {
		str += "[" + to_string(p->layerSizes[i]) + "]";
	}
	str += "[" + to_string(GetOutputSize()) + "]";
	str += "{";
	for (int j = 0; j < GetDepth(); j++) {
		str += WriteActivation(p->layerActivations[j]);
		if (j < GetDepth() - 1) {
			str += ",";
		}
	}
	str += "}";
	return str;
}