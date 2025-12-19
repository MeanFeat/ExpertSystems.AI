#pragma once
#ifndef TYPES_H
#define TYPES_H

/**
 * Activation function types supported by the neural network
 */
enum Activation {
	Linear,		// f(x) = x
	Sigmoid,	// f(x) = 1 / (1 + e^-x)
	Tanh,		// f(x) = tanh(x)
	ReLU,		// f(x) = max(0, x)
	LReLU,		// f(x) = x if x > 0, else 0.01*x (Leaky ReLU)
	Sine		// f(x) = sin(x)
};

#endif