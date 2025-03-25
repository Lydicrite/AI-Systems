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
    oss << "\n" << indent << "\t\t\t2." << featureIndex + 1 << ") ������ G ��� �������� \""
        << dataset.GetColumnHeader(featureIndex) << "\": ";

    size_t totalRows = dataset.RowCount();

    // �������� ������������� ������� ��� ������� �������� ��������
    auto classDist = dataset.GetClassDistributionForFeature(featureIndex);

    // �������� ��������
    double featureEntropy = 0.0;

    // ��� ������� �������� �������� ���������� ��������
    for (const auto& [featureValue, targetCounts] : classDist) {
        size_t totalVCount = 0;

        oss << "\n" << indent << "\t\t\t\t * �������� \"" << featureValue << "\": ";

        // ����������� ��������� �������� �������� �������� ��� �������� featureValue �������� �������� (������� �� featureIndex)
        for (const auto& [_, count] : targetCounts) {
            totalVCount += count;
        }

        // ������ �������� ��� ������������
        double featureValueEntropy = 0.0;
        for (const auto& [targetValue, count] : targetCounts) {
            double p = static_cast<double>(count) / totalVCount;

            oss << "\n" << indent << "\t\t\t\t\t <> ����������� �������� ����� \""
                << dataset.GetTargetColumnHeader() << "\" == \"" << targetValue
                << "\": pm = " << p;

            double addition = 0.0;

            if (p > 0) {
                addition = -p * log2(p);
                featureValueEntropy += addition;
            }

            oss << "\n" << indent << "\t\t\t\t\t\t <> ����� � �������� �������� �������� ����� ������: add = -p * log2(p) = " << addition;
        }

        double prob = static_cast<double>(totalVCount) / totalRows;
        featureEntropy += prob * featureValueEntropy;
        oss << "\n" << indent << "\t\t\t\t\t <> ����������� �������� ��� ��������: p = " << prob;
        oss << "\n" << indent << "\t\t\t\t\t <> �������� ����� �������� ��������: e = " << featureValueEntropy;
    }

    double gain = totalEntropy - featureEntropy;

    oss << "\n\n" << indent << "\t\t\t   ---> �������� �������� \""
        << dataset.GetColumnHeader(featureIndex) << "\": E = " << featureEntropy;

    oss << "\n" << indent << "\t\t\t   ---> �������������� ������� �������� \""
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

        // ������ Gain i-��� ��������
        double gain = CalculateInformationGain(dataset, i, totalEntropy, oss, indent);

        // ����� �������������
        if (gain > maxGain) {
            maxGain = gain;
            bestFeature = i;
        }
    }

    oss << "\n" << indent << "\t\t   ---> ����, ������ �� ��������������� �������� �������: #"
        << bestFeature << " - \"" << dataset.GetColumnHeader(bestFeature) << "\"\n";

    return bestFeature;
}

std::unique_ptr<Node> ID3::BuildTree(const DTDataset& dataset, std::ostringstream& oss) {
    oss << "\n--------------------------------------------------- ���������� ������ ������� �� ����������� ������ ������ ---------------------------------------------------";
    size_t iter = 0;
    return BuildTreeInternal(dataset, oss, iter, "");
}

std::unique_ptr<Node> ID3::BuildTreeInternal(const DTDataset& dataset, std::ostringstream& oss, size_t& iteration, const std::string& indent) {
    iteration += 1;

    oss << "\n" << indent << "\t�������� #" << iteration << ": ";

    // ������� ������
    // ������� 1: ��� ������� ����������� ������ �������� �������� ��������
    if (AllSameTargetValue(dataset)) {
        oss << "\n" << indent << "\t\t3) ������ \"���������� ����\" � ����� � ���, ��� ��� ������ ����� � ������ �������� �������� ��������\n\n\n";
        return std::make_unique<LeafNode>(dataset.GetClassDistribution().begin()->first);
    }

    // ������� 2: ��� ��������� ��� ��������� (������� ������ �������)
    if (dataset.ColumnCount() <= 1) { // ���������, ��� ������� ������� �� ���������
        return std::make_unique<LeafNode>("(������������)");
    }

    // �������� ����� ������ ������
    double totalEntropy = dataset.CalculateEntropy();
    oss << "\n" << indent << "\t\t1) ����� �������� ������ �� �������� �������� \"" << dataset.GetTargetColumnHeader() << "\": " << totalEntropy;

    // ����� ������� �������� � ������������ "���� �������"
    oss << "\n" << indent << "\t\t2) ����� ���������� �������� � ���������� �������������� ��������� G: ";
    size_t bestFeature = FindBestFeature(dataset, totalEntropy, oss, indent);
    std::string bestFeatureName = dataset.GetHeaders()[bestFeature];
    oss << "\n" << indent << "\t\t3) ������ \"���� �������\" �� ����� ��������\n\n\n";
    auto node = std::make_unique<DecisionNode>(bestFeatureName);

    // ����� ���������� �������� ������� �������� � �� ����������
    auto uniqueValues = dataset.GetUniqueValues(bestFeature);
    std::vector<std::string> sortedValues(uniqueValues.begin(), uniqueValues.end());
    std::sort(sortedValues.begin(), sortedValues.end());

    // ���������� ����������� ��� ������� �� �������� ������� ��������
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