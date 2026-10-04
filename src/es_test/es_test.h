#pragma once
#include "es_test_pch.h"
#include "stdNet.h"
#include "stdNetTrainer.h"
#include "d_NetTrainer.h"
#include "d_Matrix.h"
#include "d_math.h"

#define GENERATED_TESTS "tests_cpp.generated"
#define GENERATED_UNIT_TESTS "tests_unit.generated"
#define GENERATED_GTEST_TESTS "tests_gtest.generated"
#define DEFAULTTEXTCOLOUR 10
#define TEXTCOLOUR(token, c) HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE); SetConsoleTextAttribute(h, c); token; SetConsoleTextAttribute(h, DEFAULTTEXTCOLOUR);

static int verbosity = 3;
static const float thresholdMultiplier = FLT_EPSILON * 2.f;

struct testResult {
	bool passed;
	std::string message;
	float error;
	float threshold;
	testResult() {}
	testResult(bool p, std::string m) : passed(p), message(m) {}
	~testResult() {}
};
struct testData {
	Eigen::MatrixXf host;
	d_Matrix device;
	testData() {}
	testData(int m, int k) {
		host = Eigen::MatrixXf::Random(m, k);
		device = d_NetTrainer::to_device(host);
	}
	~testData() {
		device.free();
	}
};

void PrintHeader(std::string testType);
testResult GetOutcome(float cSum, float tSum, float thresh);
testResult testMultipy(int m, int n, int k);
testResult testTransposeRight(int m, int n, int k);
testResult testTransposeLeft(int m, int n, int k);
testResult testSum(int m, int k);
testResult testTranspose(int m, int k);
testResult testMultScalar(int m, int k);
testResult testSet(int m, int k, float val);
testResult testAdd(int m, int k);
testResult testSubtract(int m, int k);
testResult testMultElem(int m, int k);
testResult testSumRows(int m, int k);
testResult testSquare(int m, int k);
testResult testSigmoid(int m, int k);
testResult testTanh(int m, int k);
testResult testReLU(int m, int k);
testResult testLReLU(int m, int k);
testResult testSine(int m, int k);
testResult testBackProp(Net &nn, int dataCount);
testResult testCalcCost(Net &nn, int dataCount);
testResult testForward(Net &nn, int dataCount);
testResult testForwardTrain(Net &nn, int dataCount);
