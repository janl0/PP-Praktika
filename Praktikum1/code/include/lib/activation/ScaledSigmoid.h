#pragma once

// #include <activation/ScaledSigmoid.h>
#include "ActivationFunction.h"

#include <string>

namespace dnn {
// Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
class ScaledSigmoid : public ActivationFunction {
public:
    double activation(double x) override;
    double activationBackwards(double x) override;
    std::string getName() const override;
    ScaledSigmoid(double min, double max);
    double getMin() const;
    double getMax() const;

private:
    double min;
    double max;
};

}