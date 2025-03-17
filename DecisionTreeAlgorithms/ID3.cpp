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
    size_t totalRows = dataset.RowCount();

    // Получаем распределение классов для каждого значения признака
    auto classDist = dataset.GetClassDistributionForFeature(featureIndex);

    double weightedEntropy = 0.0;
    std::cout << "\n[Обработка признака] " << dataset.GetHeaders()[featureIndex]
        << "\nОбщая энтропия: " << totalEntropy << std::endl;

    for (const auto& [featureValue, targetCounts] : classDist) {
        size_t totalVCount = 0;
        for (const auto& [_, count] : targetCounts) {
            totalVCount += count;
        } 

        // Расчёт энтропии для подмножества
        double subsetEntropy = 0.0;
        for (const auto& [targetValue, count] : targetCounts) {
            double p = static_cast<double>(count) / totalVCount;
            if (p > 0) subsetEntropy -= p * log2(p);
        }

        double prob = static_cast<double>(totalVCount) / totalRows;
        weightedEntropy += prob * subsetEntropy;

        // Вывод информации
        std::cout << "  * Значение: " << featureValue
            << " | Примеров: " << totalVCount
            << " | Вероятность: " << prob
            << " | Энтропия: " << subsetEntropy << std::endl;
    }

    double gain = totalEntropy - weightedEntropy;
    std::cout << "Взвешенная энтропия: " << weightedEntropy
        << "\nИнформационный прирост: " << gain << "\n" << std::endl;

    return gain;
}

size_t ID3::FindBestFeature(const DTDataset& dataset) {
    size_t bestFeature = 0;
    double maxGain = -1.0;
    size_t targetCol = dataset.GetTargetColumn();

    for (size_t i = 0; i < dataset.ColumnCount(); ++i) {
        if (i == targetCol) 
            continue;

        double gain = CalculateInformationGain(dataset, i);
        if (gain > maxGain) {
            maxGain = gain;
            bestFeature = i;
        }
    }

    return bestFeature;
}

std::unique_ptr<Node> ID3::BuildTree(const DTDataset& dataset) {
    return BuildTreeInternal(dataset, 1);
}

std::unique_ptr<Node> ID3::BuildTreeInternal(const DTDataset& dataset, int childNumber) {
    // Условие 1: Все примеры принадлежат одному классу
    if (AllSameClass(dataset)) {
        return std::make_unique<LeafNode>(dataset.GetClassDistribution().begin()->first);
    }

    // Условие 2: Нет признаков для разбиения (остался только целевой)
    if (dataset.ColumnCount() <= 1) { // Учитываем, что целевой столбец не удаляется
        return std::make_unique<LeafNode>("(неопределено)");
    }

    size_t bestFeature = FindBestFeature(dataset);
    std::string bestFeatureName = dataset.GetHeaders()[bestFeature];
    auto node = std::make_unique<DecisionNode>(bestFeatureName);

    auto uniqueValues = dataset.GetUniqueValues(bestFeature);
    std::vector<std::string> sortedValues(uniqueValues.begin(), uniqueValues.end());
    std::sort(sortedValues.begin(), sortedValues.end()); // Сортируем для стабильности

    int childNum = 1;
    for (const auto& value : sortedValues) {
        try {
            DTDataset subset = dataset.GetFeatureValueSubset(bestFeature, value);
            subset.SetTargetColumn((dataset.GetTargetColumn() > bestFeature) ? dataset.GetTargetColumn() - 1 : dataset.GetTargetColumn());
            auto child = BuildTreeInternal(subset, childNum);
            node->AddChild(value, std::move(child));
            childNum++;
        }
        catch (const std::invalid_argument&) {
            auto classDist = dataset.GetClassDistribution();
            node->AddChild(value, std::make_unique<LeafNode>(classDist.begin()->first));
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
    tree.SetTargetColumn(dataset.GetTargetColumn());
    return tree;
}