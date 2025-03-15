#include "ID3.h"

bool ID3::AllSameClass(const DTDataset& dataset) {
    auto unique = dataset.GetUniqueValues(dataset.GetTargetColumn());
    return unique.size() == 1;
}

double ID3::CalculateInformationGain
(
    const DTDataset& dataset,
    size_t featureIndex
) {
    double totalEntropy = dataset.CalculateEntropy();
    auto uniqueValues = dataset.GetUniqueValues(featureIndex);
    double weightedEntropy = 0.0;

    for (const auto& value : uniqueValues) {
        DTDataset subset = dataset.GetFeatureValueSubset(featureIndex, value);
        double prob = static_cast<double>(subset.RowCount()) / dataset.RowCount();
        weightedEntropy += prob * subset.CalculateEntropy();
    }

    return totalEntropy - weightedEntropy;
}

size_t ID3::FindBestFeature(const DTDataset& dataset) {
    size_t bestFeature = 0;
    double maxGain = -1.0;
    size_t targetCol = dataset.GetTargetColumn();

    for (size_t i = 0; i < dataset.ColumnCount(); ++i) {
        if (i == targetCol) continue; // Пропускаем целевой столбец
        double gain = CalculateInformationGain(dataset, i);
        if (gain > maxGain) {
            maxGain = gain;
            bestFeature = i;
        }
    }

    return bestFeature;
}

std::unique_ptr<Node> ID3::BuildTree(const DTDataset& dataset) {
    return BuildTreeInternal(dataset, "", 1);
}

std::unique_ptr<Node> ID3::BuildTreeInternal(const DTDataset& dataset, const std::string& parentIndex, int childNumber) {
    std::string currentIndex;
    if (parentIndex.empty()) {
        currentIndex = std::to_string(childNumber);
    }
    else {
        currentIndex = parentIndex + "." + std::to_string(childNumber);
    }

    // Условие 1: Все примеры принадлежат одному классу
    if (AllSameClass(dataset)) {
        return std::make_unique<LeafNode>(dataset.GetClassDistribution().begin()->first, currentIndex);
    }

    // Условие 2: Нет признаков для разбиения (остался только целевой)
    if (dataset.ColumnCount() <= 1) { // Учитываем, что целевой столбец не удаляется
        return std::make_unique<LeafNode>("(неопределено)", currentIndex);
    }

    size_t bestFeature = FindBestFeature(dataset);
    std::string bestFeatureName = dataset.GetHeaders()[bestFeature];
    auto node = std::make_unique<DecisionNode>(bestFeatureName, currentIndex);

    auto uniqueValues = dataset.GetUniqueValues(bestFeature);
    std::vector<std::string> sortedValues(uniqueValues.begin(), uniqueValues.end());
    std::sort(sortedValues.begin(), sortedValues.end()); // Сортируем для стабильности

    int childNum = 1;
    for (const auto& value : sortedValues) {
        try {
            DTDataset subset = dataset.GetFeatureValueSubset(bestFeature, value);
            subset.SetTargetColumn((dataset.GetTargetColumn() > bestFeature) ? dataset.GetTargetColumn() - 1 : dataset.GetTargetColumn());
            auto child = BuildTreeInternal(subset, currentIndex, childNum);
            node->AddChild(value, std::move(child));
            childNum++;
        }
        catch (const std::invalid_argument&) {
            auto classDist = dataset.GetClassDistribution();
            node->AddChild(value, std::make_unique<LeafNode>(classDist.begin()->first, currentIndex + "." + std::to_string(childNum)));
            childNum++;
        }
    }

    return node;
}

DecisionTree ID3::Train(const DTDataset& dataset) {
    DecisionTree tree;
    tree.SetHeaders(dataset.GetHeaders());
    auto root = BuildTree(dataset);
    tree.SetRoot(std::move(root));
    return tree;
}