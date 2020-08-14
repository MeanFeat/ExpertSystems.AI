#pragma once
#include "stdNet.h"
#include "d_Matrix.h"
#include "d_math.h"
#include <Eigen/dense>
#include <iostream>

#define GENERATED_TESTS "tests_cpp.generated"
#define GENERATED_UNIT_TESTS "tests_unit.generated"

using namespace std;
using namespace Eigen;

static int verbosity = 3;
static const float thresholdMultiplier = (FLT_EPSILON) * 2.f;

struct testResult {
	bool passed;
	string message;
	testResult() {}
	testResult(bool p, string m) : passed(p), message(m) {}
	~testResult(){}
};

d_Matrix to_device(MatrixXf matrix);
MatrixXf to_host(d_Matrix d_matrix);
void PrintHeader(string testType);
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
testResult testSquare(int m, int k);
testResult testSigmoid(int m, int k);
testResult testTanh(int m, int k);
testResult testReLU(int m, int k);
testResult testLReLU(int m, int k);