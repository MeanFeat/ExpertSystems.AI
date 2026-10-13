#include "es_core_pch.h"
#include "stdNet.h"
#include "es_parser.h"
#include <stdexcept>

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
	const bool IsJson = fName.find(".json") != string::npos;
	if (IsJson)
	{
		LoadNetworkJson(fName);
	}
	else
	{
		LoadNetwork(fName);
	}
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

void Net::SaveNetworkJson(const string& fName) const
{
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

void Net::SaveNetwork(const string& fName) const
{
	if (fName.empty())
		throw invalid_argument("Network file name is empty");
	ofstream file(fName, ios::binary);
	if (!file) {
		throw runtime_error("Unable to open network file for writing: " + fName);
	}
	const int depth = GetDepth();
	if (depth <= 0 || params.W.size() != static_cast<size_t>(depth) ||
		params.b.size() != static_cast<size_t>(depth) ||
		params.layerActivations.size() != static_cast<size_t>(depth) ||
		params.layerSizes.size() != static_cast<size_t>(depth + 1)) {
		throw runtime_error("Network parameters are inconsistent and cannot be saved");
	}
	file.write((char*)&depth, sizeof(int));
	for (int lyrIdx = 0; lyrIdx < depth; lyrIdx++) {
		const MatrixXf& weight = params.W[lyrIdx];
		const MatrixXf& bias = params.b[lyrIdx];
		if (weight.rows() <= 0 || weight.cols() <= 0 ||
			bias.rows() != weight.rows() || bias.cols() != 1 ||
			params.layerSizes[lyrIdx] != weight.cols() ||
			params.layerSizes[lyrIdx + 1] != weight.rows() ||
			(lyrIdx > 0 && weight.cols() != params.W[lyrIdx - 1].rows())) {
			throw runtime_error("Network layer dimensions are inconsistent and cannot be saved");
		}
		int rows = static_cast<int>(weight.rows());
		int cols = static_cast<int>(weight.cols());
		file.write((char*)&rows, sizeof(int));
		file.write((char*)&cols, sizeof(int));
		file.write((char*)weight.data(), static_cast<streamsize>(weight.size()) * sizeof(float));

		rows = static_cast<int>(bias.rows());
		cols = static_cast<int>(bias.cols());
		file.write((char*)&rows, sizeof(int));
		file.write((char*)&cols, sizeof(int));
		file.write((char*)bias.data(), static_cast<streamsize>(bias.size()) * sizeof(float));

		int act = static_cast<int>(params.layerActivations[lyrIdx]);
		file.write((char*)&act, sizeof(int));
	}
	file.close();
	if (!file) {
		throw runtime_error("Failed while writing network file: " + fName);
	}
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
		throw invalid_argument("Network file name is empty");
	ifstream file(fName, ios::binary);
	if (!file) {
		throw runtime_error("Unable to open network file for reading: " + fName);
	}
	file.seekg(0, ios::end);
	const streamoff fileSize = file.tellg();
	file.seekg(0, ios::beg);
	if (!file || fileSize < static_cast<streamoff>(sizeof(int))) {
		throw runtime_error("Invalid binary network file: " + fName);
	}
	int depth = 0;
	file.read((char*)&depth, sizeof(int));
	constexpr streamoff minimumLayerSize = 5 * static_cast<streamoff>(sizeof(int)) + 2 * static_cast<streamoff>(sizeof(float));
	if (!file || depth <= 0 ||
		static_cast<streamoff>(depth) > (fileSize - static_cast<streamoff>(sizeof(int))) / minimumLayerSize) {
		throw runtime_error("Invalid binary network file: " + fName);
	}

	vector<MatrixXf> weights;
	vector<MatrixXf> biases;
	vector<Activation> activations;
	vector<int> layerSizes;
	weights.reserve(depth);
	biases.reserve(depth);
	activations.reserve(depth);
	layerSizes.reserve(depth + 1);

	for (int lyrIdx = 0; lyrIdx < depth; lyrIdx++) {
		int rows = 0, cols = 0;
		file.read((char*)&rows, sizeof(int));
		file.read((char*)&cols, sizeof(int));
		if (!file || rows <= 0 || cols <= 0) {
			throw runtime_error("Invalid weight dimensions in binary network file: " + fName);
		}
		const streamoff weightDataPosition = file.tellg();
		const uint64_t weightElementCount = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
		if (weightDataPosition < 0 || weightDataPosition > fileSize || weightElementCount >
			static_cast<uint64_t>((fileSize - weightDataPosition) / static_cast<streamoff>(sizeof(float)))) {
			throw runtime_error("Invalid weight data in binary network file: " + fName);
		}
		MatrixXf w(rows, cols);
		file.read((char*)w.data(), static_cast<streamsize>(weightElementCount) * sizeof(float));
		if (!file) {
			throw runtime_error("Truncated weights in binary network file: " + fName);
		}

		file.read((char*)&rows, sizeof(int));
		file.read((char*)&cols, sizeof(int));
		if (!file || rows != w.rows() || cols != 1) {
			throw runtime_error("Invalid bias dimensions in binary network file: " + fName);
		}
		const streamoff biasDataPosition = file.tellg();
		if (biasDataPosition < 0 || biasDataPosition > fileSize || static_cast<uint64_t>(rows) >
			static_cast<uint64_t>((fileSize - biasDataPosition) / static_cast<streamoff>(sizeof(float)))) {
			throw runtime_error("Invalid bias data in binary network file: " + fName);
		}
		MatrixXf b(rows, cols);
		file.read((char*)b.data(), static_cast<streamsize>(rows) * cols * sizeof(float));
		if (!file) {
			throw runtime_error("Truncated biases in binary network file: " + fName);
		}

		int act = 0;
		file.read((char*)&act, sizeof(int));
		if (!file || act < Linear || act > Sine) {
			throw runtime_error("Invalid activation in binary network file: " + fName);
		}

		if (lyrIdx == 0) {
			layerSizes.push_back(static_cast<int>(w.cols()));
		}
		else if (w.cols() != weights.back().rows()) {
			throw runtime_error("Inconsistent layer dimensions in binary network file: " + fName);
		}
		layerSizes.push_back(static_cast<int>(w.rows()));
		weights.push_back(w);
		biases.push_back(b);
		activations.push_back(static_cast<Activation>(act));
	}
	if (file.peek() != char_traits<char>::eof()) {
		throw runtime_error("Unexpected trailing data in binary network file: " + fName);
	}

	params.layerSizes.swap(layerSizes);
	params.W.swap(weights);
	params.b.swap(biases);
	params.layerActivations.swap(activations);
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

string Net::ToString() const
{
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
