#include <activation/Sigmoid.h>

#include <cmath>

double dnn::Sigmoid::activation(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double dnn::Sigmoid::activationBackwards(double x) {
	return activation(x) * (1 - activation(x));
}

std::string dnn::Sigmoid::getName() {
	return "sigmoid";
}