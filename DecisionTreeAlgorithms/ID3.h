#pragma once
#include "DTDataset.h"
#include "DecisionTree.h"

class ID3 {
private:
    static bool AllSameClass(const DTDataset& dataset);
    static double CalculateInformationGain(const DTDataset& dataset, size_t featureIndex);
    static size_t FindBestFeature(const DTDataset& dataset);
    static std::unique_ptr<Node> BuildTreeInternal(const DTDataset& dataset, const std::string& parentIndex, int childNumber);

    DTDataset _trainDataset;
    std::vector<std::string> _originalHeaders;

public:
    ID3(const DTDataset& dataset) : _trainDataset(dataset), _originalHeaders(dataset.GetHeaders()) { }

    static std::unique_ptr<Node> BuildTree(const DTDataset& dataset);
    static DecisionTree Train(const DTDataset& dataset);
};

