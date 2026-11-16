#include <activation/ScaledSigmoid.h>

#include <cmath>

dnn::ScaledSigmoid::ScaledSigmoid(double min, double max) {
	this->min = min;
	this->max = max;
}

double dnn::ScaledSigmoid::getMin() const {
	return this->min;
}

double dnn::ScaledSigmoid::getMax() const {
	return this->max;
}

double dnn::ScaledSigmoid::activation(double x) {
	return min + (max-min) * (1.0 / (1.0 + exp(-x)));
}

double dnn::ScaledSigmoid::activationBackwards(double x) {
	return (max - min) * (1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))));
}

std::string dnn::ScaledSigmoid::getName() {
	return "scaled_sigmoid";
}