#include <activation/ScaledSigmoid.h>

#include <cmath>

namespace dnn {

ScaledSigmoid::ScaledSigmoid(double min, double max) {
	this->min = min;
	this->max = max;
}

double ScaledSigmoid::getMin() const {
	return this->min;
}

double ScaledSigmoid::getMax() const {
	return this->max;
}

double ScaledSigmoid::activation(double x) {
	return min + (max-min) * (1.0 / (1.0 + exp(-x)));
}

double caledSigmoid::activationBackwards(double x) {
	return (max - min) * (1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))));
}

std::string ScaledSigmoid::getName() {
	return "scaled_sigmoid";
}

}