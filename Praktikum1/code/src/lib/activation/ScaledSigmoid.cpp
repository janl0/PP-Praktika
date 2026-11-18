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
	return this->min + (this->max-this->min) * (1.0 / (1.0 + exp(-x)));
}

double ScaledSigmoid::activationBackwards(double x) {
	return (this->max - this->min) * (1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))));
}

std::string ScaledSigmoid::getName() const {
	return "scaled_sigmoid";
}

}