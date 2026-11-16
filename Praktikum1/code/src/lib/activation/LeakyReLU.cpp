#include <activation/LeakyReLU.h>

double dnn::LeakyReLU::activation(double x) {
	if x > 0 {
		return x;
	}
	else {
		return 0.01 * x;
	}
}

double dnn::LeakyReLU::activationBackwards(double x) {
	if x > 0 {
		return 1.0;
	}
	else {
		return 0.01;
	}
}

std::string dnn::LeakyReLU::getName() {
	return "leaky_relu";
}