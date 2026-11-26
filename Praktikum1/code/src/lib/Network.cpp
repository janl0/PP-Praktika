#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {

Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters, LossFunction* lossFunction) {
    this->ownedLossFunction = lossFunction;
    
    // after this push back is O(1) without reallocation
    this->layers.reserve(layerParameters.size());

    for (int i = 0; i < layerParameters.size(); i++) {
        std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*> layerParameter = layerParameters[i];
        Layer* layer = new Layer(i, 0, std::get<0>(layerParameter)[0].size(), std::get<0>(layerParameter).size(), std::get<2>(layerParameter));
        layer->setParameters(std::get<0>(layerParameter), std::get<1>(layerParameter));
        this->layers.push_back(layer);
    }
}

Network::~Network() {
    delete this->ownedLossFunction;

    for (Layer* layer : this->layers) {
        delete layer;
    }
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return this->ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return this->output;
}

const std::vector<double>& Network::getCostHistory() const {
    return this->costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return this->accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return this->layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : this->layers) {
        delete layer;
    }
    this->layers = std::move(newLayers);
    this->initialInput.clear();
    this->output.clear();
    this->costHistory.clear();
    this->accuracyHistory.clear();
}

Network& Network::addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction) {
    Layer* layer = new Layer(layers.size(), nextSeed++, inputDim, outputDim, activationFunction);
    layer->init();
    this->layers.push_back(layer);
    return *this;
}

const std::vector<double>& Network::getLayerInput(uint64_t layerId) const {
    if (layerId == 0) {
        return this->initialInput;
    }
    else {
        return this->layers[layerId - 1]->getActivationOutput();
    }
}

void Network::fullForwardPropagation(const std::vector<double>& input) {
    this->initialInput = input;

    for (auto it = this->layers.begin(); it != this->layers.end(); it++) {
        (*it)->forwardPropagation(this->getLayerInput((*it)->getId()));
    }

    if (this->layers.size() > 0) {
        this->output = this->layers.back()->getActivationOutput();
    }
    else {
        this->output = input;
    }
}

void Network::fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) {
    if (this->ownedLossFunction == nullptr) {
        throw std::logic_error("no loss function");
    }

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

    for (Layer* layer : this->layers) {
        layer->commitToWeights(learningRate);
    }

    this->costHistory.clear();
    this->accuracyHistory.clear();
    for (int i = 0; i < epochs; i++) {
        double epochCost = 0.0;
        double epochAccuracy = 0.0;

        for (int j = 0; j < inputs.size(); j++) {

            std::vector<double> shortedInput(inputs[j].cbegin(), inputs[j].cbegin() + this->layers[0]->getWeights()[0].size());
            std::vector<double> shortedOutput(actualOutputs[j].cbegin(), actualOutputs[j].cbegin() + this->layers[0]->getWeights().size());

            this->fullForwardPropagation(shortedInput);

            epochCost += this->ownedLossFunction->getCostValue(this->output, shortedOutput);
            epochAccuracy += this->ownedLossFunction->getAccuracyValue(this->output, shortedOutput);

            this->fullBackwardPropagation(this->output, shortedOutput);

            if (j % batchSize == 0) {
                for (Layer* layer : this->layers) {
                    layer->commitToWeights(learningRate);
                }
            }
        }
        for (Layer* layer : this->layers) {
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

        std::vector<double> shortedInput(inputs[i].cbegin(), inputs[i].cbegin() + this->layers[0]->getWeights()[0].size());
        std::vector<double> shortedOutput(actualOutputs[i].cbegin(), actualOutputs[i].cbegin() + this->layers[0]->getWeights().size());

        this->fullForwardPropagation(shortedInput);

        totalCorrect += this->ownedLossFunction->getAccuracyValue(this->output, shortedOutput);
    }

    return totalCorrect / inputs.size();
}

}
