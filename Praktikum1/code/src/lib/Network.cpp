#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {

Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters, LossFunction* lossFunction) {
    this->ownedLossFunction = lossFunction;

    for (int i = 0; i < layerParameters.size(); i++) {
        std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*> layerParameter = layerParameters.at(i);
        Layer layer(i, 0, std::get<0>(layerParameter).at(0).size(), std::get<0>(layerParameter).size(), std::get<2>(layerParameter));
        layer.setParameters(std::get<0>(layerParameter), std::get<1>(layerParameter));
        layers.push_back(&layer);
    }
}

Network::~Network() {
    delete this->ownedLossFunction;

    for (Layer* layer : layers) {
        delete layer;
    }
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
    Layer layer(0, nextSeed++, inputDim, outputDim, activationFunction);
    return *this;
}

const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
    if (layerId == 0) {
        return this->initialInput;
    }
    else {
        return this->layers.at(layerId - 1)->getActivationOutput();
    }
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    this->initialInput = input;

    for (auto it = this->layers.begin(); it != this->layers.end(); it++) {
        (*it)->forwardPropagation(this->getLayerInput((*it)->getId()));
    }

    if (layers.size() > 0) {
        output = this->layers.back()->getActivationOutput();
    }
}

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) {
    std::vector<double> dACurr = this->ownedLossFunction->getGradient(predictedOutput, actualOutput);

    for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
        (*it)->backwardPropagation(dACurr, this->getLayerInput((*it)->getId()));
        dACurr = (*it)->getDPreviousLayerOutput();
    }
}

void Network::trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
    int epochs, double learningRate) {
    if (this->ownedLossFunction == nullptr) {
        throw std::logic_error("no loss function");
    }

    if (inputs.size() == 0) {
        throw std::invalid_argument("inputs has to be non-empty");
    }

    if (actualOutputs.size() == 0) {
        throw std::invalid_argument("actualOutputs has to be non-empty");
    }

    if (inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("length of inputs does not match the length of actualOutputs");
    }

    if (batchSize <= 0) {
        throw std::invalid_argument("batchSize has to be positive");
    }

    this->costHistory.clear();
    this->accuracyHistory.clear();
    for (int i = 0; i < epochs; i++) {
        double epochCost = 0.0;
        double epochAccuracy = 0.0;

        for (int j = 0; j < inputs.size(); j++) {
            if (j % batchSize == 0 && j != 0) {
                for (Layer* layer : layers) {
                    layer->commitToWeights(learningRate);
                }
            }

            this->fullForwardPropagation(inputs.at(j));

            epochCost += this->ownedLossFunction->getCostValue(this->output, actualOutputs.at(j));
            epochAccuracy += this->ownedLossFunction->getAccuracyValue(this->output, actualOutputs.at(j));

            this->fullBackwardPropagation(this->output, actualOutputs.at(j));
        }
        for (Layer* layer : layers) {
            layer->commitToWeights(learningRate);
        }

        this->costHistory.push_back(epochCost / inputs.size());
        this->accuracyHistory.push_back(epochAccuracy / inputs.size());
    }
}

double Network::computeAccuracy(const std::vector<std::vector<double>>&inputs, const std::vector<std::vector<double>>&actualOutputs) {
    if (this->ownedLossFunction == nullptr) {
        throw std::logic_error("no loss function");
    }

    if (inputs.size() == 0) {
        throw std::invalid_argument("inputs has to be non-empty");
    }

    if (actualOutputs.size() == 0) {
        throw std::invalid_argument("actualOutputs has to be non-empty");
    }

    if (inputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("length of inputs does not match the length of actualOutputs");
    }

    double totalCorrect = 0.0;

    for (int i = 0; i < inputs.size(); i++) {
        this->fullForwardPropagation(inputs.at(i));

        totalCorrect += this->ownedLossFunction->getAccuracyValue(this->output, actualOutputs.at(i));
    }

    return totalCorrect / inputs.size();
}

}
