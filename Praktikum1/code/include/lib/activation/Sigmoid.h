#pragma once

// #include <activation/ScaledSigmoid.h>
#include "ActivationFunction.h"

#include <string>

namespace dnn {

class Sigmoid : public ActivationFunction {
public:
    double activation(double x) override;
    double activationBackwards(double x) override;
    std::string getName() const override;
};

}