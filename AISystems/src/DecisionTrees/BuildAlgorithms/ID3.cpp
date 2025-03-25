#include <../include/DecisionTrees/BuildAlgorithms/ID3.h>

bool ID3::AllSameTargetValue(const DTDataset& dataset) {
    auto unique = dataset.GetUniqueValues(dataset.GetTargetColumn());
    return unique.size() == 1;
}

double ID3::CalculateInformationGain
(
    const DTDataset& dataset,
    size_t featureIndex,
    const double& totalEntropy,
    std::ostringstream& oss,
    const std::string& indent
) {
    oss << "\n" << indent << "\t\t\t2." << featureIndex + 1 << ") Расчёт G для признака \""
        << dataset.GetColumnHeader(featureIndex) << "\": ";

    size_t totalRows = dataset.RowCount();

    // Получаем распределение классов для каждого значения признака
    auto classDist = dataset.GetClassDistributionForFeature(featureIndex);

    // Энтропия признака
    double featureEntropy = 0.0;

    // Для каждого значения текущего нецелевого признака
    for (const auto& [featureValue, targetCounts] : classDist) {
        size_t totalVCount = 0;

        oss << "\n" << indent << "\t\t\t\t * значение \"" << featureValue << "\": ";

        // Вероятность встретить значение целевого признака при значении featureValue текущего признака (который по featureIndex)
        for (const auto& [_, count] : targetCounts) {
            totalVCount += count;
        }

        // Расчёт энтропии для подмножества
        double featureValueEntropy = 0.0;
        for (const auto& [targetValue, count] : targetCounts) {
            double p = static_cast<double>(count) / totalVCount;

            oss << "\n" << indent << "\t\t\t\t\t <> вероятность получить исход \""
                << dataset.GetTargetColumnHeader() << "\" == \"" << targetValue
                << "\": pm = " << p;

            double addition = 0.0;

            if (p > 0) {
                addition = -p * log2(p);
                featureValueEntropy += addition;
            }

            oss << "\n" << indent << "\t\t\t\t\t\t <> вклад в энтропию значения признака этого исхода: add = -p * log2(p) = " << addition;
        }

        double prob = static_cast<double>(totalVCount) / totalRows;
        featureEntropy += prob * featureValueEntropy;
        oss << "\n" << indent << "\t\t\t\t\t <> вероятность получить это значение: p = " << prob;
        oss << "\n" << indent << "\t\t\t\t\t <> энтропия этого значения признака: e = " << featureValueEntropy;
    }

    double gain = totalEntropy - featureEntropy;

    oss << "\n\n" << indent << "\t\t\t   ---> Энтропия признака \""
        << dataset.GetColumnHeader(featureIndex) << "\": E = " << featureEntropy;

    oss << "\n" << indent << "\t\t\t   ---> Информационный прирост признака \""
        << dataset.GetColumnHeader(featureIndex) << "\": G = " << gain;

    return gain;
}


size_t ID3::FindBestFeature(const DTDataset& dataset, const double& totalEntropy, std::ostringstream& oss, const std::string& indent) {
    size_t bestFeature = 0;
    double maxGain = -1.0;
    size_t targetCol = dataset.GetTargetColumn();

    for (size_t i = 0; i < dataset.ColumnCount(); ++i) {
        if (i == targetCol)
            continue;

        // Расчёт Gain i-ого признака
        double gain = CalculateInformationGain(dataset, i, totalEntropy, oss, indent);

        // Поиск максимального
        if (gain > maxGain) {
            maxGain = gain;
            bestFeature = i;
        }
    }

    oss << "\n" << indent << "\t\t   ---> итак, лучший по информационному приросту признак: #"
        << bestFeature << " - \"" << dataset.GetColumnHeader(bestFeature) << "\"\n";

    return bestFeature;
}

std::unique_ptr<Node> ID3::BuildTree(const DTDataset& dataset, std::ostringstream& oss) {
    oss << "\n--------------------------------------------------- Построение дерева решения по переданному набору данных ---------------------------------------------------";
    size_t iter = 0;
    return BuildTreeInternal(dataset, oss, iter, "");
}

std::unique_ptr<Node> ID3::BuildTreeInternal(const DTDataset& dataset, std::ostringstream& oss, size_t& iteration, const std::string& indent) {
    iteration += 1;

    oss << "\n" << indent << "\tИтерация #" << iteration << ": ";

    // Условия выхода
    // Условие 1: Все примеры принадлежат одному значению целевого признака
    if (AllSameTargetValue(dataset)) {
        oss << "\n" << indent << "\t\t3) Создаём \"замыкающий узел\" в связи с тем, что все исходы ведут к одному значению целевого признака\n\n\n";
        return std::make_unique<LeafNode>(dataset.GetClassDistribution().begin()->first);
    }

    // Условие 2: Нет признаков для разбиения (остался только целевой)
    if (dataset.ColumnCount() <= 1) { // Учитываем, что целевой столбец не удаляется
        return std::make_unique<LeafNode>("(неопределено)");
    }

    // Энтропия всего набора данных
    double totalEntropy = dataset.CalculateEntropy();
    oss << "\n" << indent << "\t\t1) Общая энтропия набора по целевому признаку \"" << dataset.GetTargetColumnHeader() << "\": " << totalEntropy;

    // Поиск лучшего признака и формирование "узла решения"
    oss << "\n" << indent << "\t\t2) Поиск нецелевого признака с наибольшим информационным приростом G: ";
    size_t bestFeature = FindBestFeature(dataset, totalEntropy, oss, indent);
    std::string bestFeatureName = dataset.GetHeaders()[bestFeature];
    oss << "\n" << indent << "\t\t3) Создаём \"узел решения\" по этому признаку\n\n\n";
    auto node = std::make_unique<DecisionNode>(bestFeatureName);

    // Поиск уникальных значений лучшего признака и их сортировка
    auto uniqueValues = dataset.GetUniqueValues(bestFeature);
    std::vector<std::string> sortedValues(uniqueValues.begin(), uniqueValues.end());
    std::sort(sortedValues.begin(), sortedValues.end());

    // Построение ответвлений для каждого из значений лучшего признака
    size_t cILength = (iteration == 1) ? 2 : iteration + 2;
    std::string childIndent(cILength, ' ');
    size_t innerCounter = 1;

    for (const auto& value : sortedValues) {
        try {
            DTDataset subset = dataset.GetFeatureValueSubset(bestFeature, value);
            subset.SetTargetColumn((dataset.GetTargetColumn() > bestFeature) ? dataset.GetTargetColumn() - 1 : dataset.GetTargetColumn());
            auto child = BuildTreeInternal(subset, oss, iteration, childIndent);
            node->AddChild(value, std::move(child));
        }
        catch (const std::invalid_argument&) {
            auto classDist = dataset.GetClassDistribution();
            node->AddChild(value, std::make_unique<LeafNode>(classDist.begin()->first));
        }
        innerCounter++;
    }

    return node;
}



DecisionTree ID3::Train(const DTDataset& dataset) {
    DecisionTree tree;
    tree.SetHeaders(dataset.GetHeaders());
    tree.SetTargetColumn(dataset.GetTargetColumn());
    tree.ClearBuildingProcessOSS();

    auto root = BuildTree(dataset, tree.GetBuildingProcessOSS());
    tree.SetRoot(std::move(root));
    return tree;
}