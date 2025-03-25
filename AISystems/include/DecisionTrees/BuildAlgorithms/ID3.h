#pragma once
#include "DecisionTrees/DecisionTree/DecisionTree.h"
#include "DecisionTrees/DTDataset.h"

class ID3 {
private:
    static bool AllSameTargetValue(const DTDataset& dataset);

    static double CalculateInformationGain
    (
        const DTDataset& dataset,
        size_t featureIndex,
        const double& totalEntropy,
        std::ostringstream& oss,
        const std::string& indent
    );

    static size_t FindBestFeature
    (
        const DTDataset& dataset,
        const double& totalEntropy,
        std::ostringstream& oss,
        const std::string& indent
    );

    static std::unique_ptr<Node> BuildTreeInternal
    (
        const DTDataset& dataset,
        std::ostringstream& oss,
        size_t& iteration,
        const std::string& indent
    );

    static std::unique_ptr<Node> BuildTree(const DTDataset& dataset, std::ostringstream& oss);

    DTDataset _trainDataset;
    std::vector<std::string> _originalHeaders;

public:
    ID3(const DTDataset& dataset)
        : _trainDataset(dataset), _originalHeaders(dataset.GetHeaders()) {
    }

    static DecisionTree Train(const DTDataset& dataset);
};

