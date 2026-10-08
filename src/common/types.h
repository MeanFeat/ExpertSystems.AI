#pragma once
#if !defined(TYPES_H)

enum Activation {
	Linear,
	Sigmoid,
	Tanh,
	ReLU,
	LReLU,
	Sine
};

enum NetBatchShuffleType {
	None,
	ShuffleRandom,
	SlideWindow
};

#define TYPES_H
#endif