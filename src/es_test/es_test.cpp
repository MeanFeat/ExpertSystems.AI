#include "es_test_pch.h"
#include "es_test.h"
using namespace std;
using namespace Eigen;
void PrintHeader(string testType) {
	if (verbosity > 0) {
		const int len = (int)strlen(testType.c_str());
		string border = "============";
		for (int i = 0; i < len; i++) {
			border += "=";
		}
		cout << border << endl;
		cout << "||Testing " << testType << "||" << endl;
		cout << border << endl;
	}
}
// Relative tolerances, set ~5-10x above the worst error measured per category; atol covers sums near zero
static constexpr float kRelMult = 5e-6f;
static constexpr float kRelSum = 1e-5f;
static constexpr float kRelElem = 1e-6f;
static constexpr float kRelTrig = 3e-5f;
static constexpr float kRelNet = 1e-5f;// Accumulate in double so the reference sum isn't limited by float summation error
static float HostSum(const MatrixXf& m) {
	return float(m.cast<double>().sum());
}
string GetOutcomeString(const float cSum, const float tSum, const float diff, const float thresh, const bool passed) {
	string out;
	if (verbosity >= 2) {
		out += "Eigen: " + to_string(cSum) + " Device: " + to_string(tSum) + "\n";
		out += "Error " + to_string(diff) + " : Threshold " + to_string(thresh) + "\n";
	}
	if (verbosity >= 1) {
		out += "======================================================>> ";
		if (passed) {
			out += "PASS!";
		}
		else {
			out += "fail... " + to_string(diff - thresh);
		}
	}
	return out;
}
testResult GetOutcome(const float cSum, const float tSum, const float rtol, const float atol) {
	testResult result;
	const float diff = abs(cSum - tSum);
	const float thresh = atol + rtol * max(abs(cSum), abs(tSum));
	result.passed = diff <= thresh;
	result.message = GetOutcomeString(cSum, tSum, diff, thresh, result.passed);
	// to_string prints 6 decimals, so anything below 5e-7 is displayed as 0.000000
	const int passCol = diff >= 5e-7f ? 14 : 10;
	TEXTCOLOUR(cout << result.message << endl; , result.passed ? passCol : 12);
	return result;
}
testResult testMultipy(const int m, const int n, const int k) {
	cout << "Testing Multiply " << m << "," << n << " * " << n << "," << k << endl;
	const testData A = testData(m, n);
	const testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult(&d_C, &A.device, &B.device);
	return GetOutcome(HostSum(A.host*B.host), HostSum(MatrixXf(d_NetTrainer::to_host(d_C))), kRelMult);
}
testResult testTransposeRight(const int m, const int n, const int k) {
	cout << "Testing Multiply (" << m << "," << n << ") * (" << n << "," << k << ").transpose()" << endl;
	const testData A = testData(m, n);
	testData B = testData(k, n);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_rhsT(&d_C, &A.device, &B.device);
	return GetOutcome(HostSum(A.host*B.host.transpose()), HostSum(d_NetTrainer::to_host(d_C)), kRelMult);
}
testResult testTransposeLeft(const int m, const int n, const int k) {
	cout << "Testing Multiply (" << m << "," << n << ").transpose() * (" << n << "," << k << ")" << endl;
	testData A = testData(n, m);
	const testData B = testData(n, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_mult_lhsT(&d_C, &A.device, &B.device);
	return GetOutcome(HostSum(A.host.transpose()*B.host), HostSum(d_NetTrainer::to_host(d_C)), kRelMult);
}
testResult testSum(const int m, const int k) {
	cout << "Testing Sum " << m << "," << k << endl;
	const testData A = testData(m, k);
	float* d_testSum;
	float result;
	cudaMalloc(&d_testSum, sizeof(float));
	d_sumMatrix(d_testSum, &A.device);
	cudaMemcpyAsync(&result, d_testSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_testSum);
	return GetOutcome(HostSum(A.host), result, kRelSum);
}
testResult testTranspose(const int m, const int k) {
	cout << "Testing Transpose " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_transpose(&d_C, &A.device);
	MatrixXf control = A.host.transpose();
	MatrixXf result = d_NetTrainer::to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		const float con = *(control.data() + i);
		const float res = *(result.data() + i);
		if (abs(con - res) > FLT_EPSILON) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testMultScalar(const int m, const int k) {
	cout << "Testing Multiply Element (" << m << "," << k << ") * b" << endl;
	testData A = testData(m, k);
	const float r = float(rand()) / float(RAND_MAX);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, k));
	d_mult_scalar(&A.device, r);
	const float controlSum = HostSum(MatrixXf(A.host * r));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelElem);
}
testResult testAdd(const int m, const int k) {
	cout << "Testing Add " << m << "," << k << " (+) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_add_elem(&d_C, A.device, B.device);
	const float controlSum = HostSum(MatrixXf(A.host.array() + B.host.array()));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(d_C))), kRelElem);
}
testResult testSubtract(const int m, const int k) {
	cout << "Testing Subtract " << m << "," << k << " (-) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_Matrix(m, k);
	d_subtract_elem(&d_C, A.device, B.device);
	return GetOutcome(HostSum(MatrixXf(A.host.array() - B.host.array())), HostSum(MatrixXf(d_NetTrainer::to_host(d_C))), kRelElem);
}
testResult testMultElem(const int m, const int k) {
	cout << "Testing MultElem " << m << "," << k << " (*) " << m << "," << k << endl;
	testData A = testData(m, k);
	testData B = testData(m, k);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, k));
	d_mult_elem(&d_C, A.device, B.device);
	return GetOutcome(HostSum(MatrixXf(A.host.array() * B.host.array())), HostSum(MatrixXf(d_NetTrainer::to_host(d_C))), kRelElem);
}
testResult testSumRows(const int m, const int k) {
	cout << "Testing SumRows " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_NetTrainer::to_device(MatrixXf::Zero(m, 1));
	d_sumRows(&d_C, &A.device);
	MatrixXf result = MatrixXf(d_NetTrainer::to_host(d_C));
	MatrixXf control = A.host.rowwise().sum();
	string elemList = "";
	for (int i = 0; i < result.size(); i++) {
		const float r = *(result.data() + i);
		const float c = *(control.data() + i);
		elemList += to_string(r - c) + ",";
	}
	return testResult(result.isApprox(A.host.rowwise().sum()), elemList);
}
testResult testSet(const int m, const int k, const float val) {
	cout << "Testing Set " << m << "," << k << " (=) " << val << endl;
	const testData A = testData(m, k);
	d_Matrix d_C = A.device;
	d_set_elem(&d_C, val);
	MatrixXf result = d_NetTrainer::to_host(d_C);
	string elemList = "";
	bool passed = true;
	for (int i = 0; i < result.size(); i++) {
		const float ith = *(result.data() + i);
		if (ith != val) {
			passed = false;
			elemList += to_string(i) + ", ";
		}
	}
	return testResult(passed, elemList);
}
testResult testSquare(const int m, const int k) {
	cout << "Testing Square " << m << "," << k << endl;
	testData A = testData(m, k);
	d_Matrix d_C = d_Matrix(k, m);
	d_square(&d_C, &A.device);
	const float controlSum = HostSum(MatrixXf(A.host.array()* A.host.array()));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(d_C))), kRelElem);
}
testResult testSigmoid(const int m, const int k) {
	cout << "Testing Sigmoid " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Sigmoid);
	const float controlSum = HostSum(MatrixXf(Net::Activate(A.host, Activation::Sigmoid)));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelElem);
}
testResult testTanh(const int m, const int k) {
	cout << "Testing Tanh " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Tanh);
	const float controlSum = HostSum(MatrixXf(Net::Activate(A.host, Activation::Tanh)));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelTrig);
}
testResult testReLU(const int m, const int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::ReLU);
	const float controlSum = HostSum(MatrixXf(Net::Activate(A.host, Activation::ReLU)));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelElem);
}
testResult testLReLU(const int m, const int k) {
	cout << "Testing ReLU " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::LReLU);
	const float controlSum = HostSum(MatrixXf(Net::Activate(A.host, Activation::LReLU)));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelElem);
}
testResult testSine(const int m, const int k) {
	cout << "Testing Sine " << m << "," << k << endl;
	testData A = testData(m, k);
	d_activate(&A.device, Activation::Sine);
	const float controlSum = HostSum(MatrixXf(Net::Activate(A.host, Activation::Sine)));
	return GetOutcome(controlSum, HostSum(MatrixXf(d_NetTrainer::to_host(A.device))), kRelTrig);
}
testResult testBackProp(Net &nn, const int dataCount) {
	const MatrixXf data = MatrixXf::Random(nn.GetInputSize(), dataCount);
	const MatrixXf labels = MatrixXf::Random(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	Net d_nn = (nn);
	NetTrainer h_trainer = NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	h_trainer.ForwardTrain();
	h_trainer.BackwardPropagation();
	const MatrixXf control = h_trainer.GetTrainParams().dW[0];
	d_NetTrainer d_trainer = d_NetTrainer(&d_nn, data, labels, 1.f, 0.25f, 0.f);
	d_trainer.ForwardTrain();
	d_trainer.BackwardPropagation();
	const MatrixXf test = d_NetTrainer::to_host(d_trainer.GetDerivatives().d_dW[0]);
	return GetOutcome(HostSum(control), HostSum(test), kRelNet);
}
testResult testCalcCost(Net &nn, const int dataCount) {
	const testData test = testData(nn.GetOutputSize(), dataCount);
	const testData labels = testData(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, test.host, labels.host, 1.f, 0.25f, 0.f);
	const float control = h_trainer.CalcCost(test.host, labels.host);
	d_NetTrainer d_trainer = d_NetTrainer(&nn, test.host, labels.host, 1.f, 0.25f, 0.f);
	const float result = d_trainer.CalcCost(test.device, labels.device);
	return GetOutcome(control, result, kRelNet);
}
testResult testForward(Net &nn, const int dataCount) {
	const testData data = testData(nn.GetInputSize(), dataCount);
	const testData labels = testData(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, data.host, labels.host, 1.f, 0.25f, 0.f);
	const d_NetTrainer d_trainer = d_NetTrainer(&nn, data.host, labels.host, 1.f, 0.25f, 0.f);
	const MatrixXf control = nn.ForwardPropagation(data.host);
	const d_Matrix result = d_trainer.Forward(data.device);
	const MatrixXf test = d_NetTrainer::to_host(result);
	return GetOutcome(HostSum(control), HostSum(test), kRelNet);
}
testResult testForwardTrain(Net &nn, const int dataCount) {
	const MatrixXf data = MatrixXf::Random(nn.GetInputSize(), dataCount);
	const MatrixXf labels = MatrixXf::Random(nn.GetOutputSize(), dataCount);
	nn.RandomInit(0.15f);
	NetTrainer h_trainer = NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	const MatrixXf control = h_trainer.ForwardTrain();
	d_NetTrainer d_trainer = d_NetTrainer(&nn, data, labels, 1.f, 0.25f, 0.f);
	d_trainer.ForwardTrain();
	const MatrixXf test = d_NetTrainer::to_host(d_trainer.GetCache().d_A.back());
	return GetOutcome(HostSum(control), HostSum(test), kRelNet);
}
testResult testTrainEpoch(Net &nn, const int dataCount) {
	const MatrixXf data = MatrixXf::Random(nn.GetInputSize(), dataCount);
	const MatrixXf labels = MatrixXf::Random(nn.GetOutputSize(), dataCount);
	const d_NetBatchShuffleType shuffleTypes[] = { None, ShuffleRandom, SlideWindow };
	bool passed = true;
	for (const d_NetBatchShuffleType shuffleType : shuffleTypes) {
		nn.RandomInit(0.15f);
		const MatrixXf initialWeights = nn.GetParams().W[0];
		d_NetTrainer trainer(&nn, data, labels, 1.f, 0.25f, 0.f, d_NetBatchParams(2, shuffleType));
		trainer.TrainSingleEpoch();
		trainer.RefreshHostNetwork();
		passed = passed && std::isfinite(trainer.GetCache().cost) &&
			!nn.GetParams().W[0].isApprox(initialWeights, 1e-7f);
	}
	return GetOutcome(passed ? 1.f : 0.f, 1.f, 0.f, 0.f);
}
