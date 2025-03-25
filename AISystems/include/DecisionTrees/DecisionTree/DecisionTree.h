#pragma once
#include <sstream>
#include "Nodes/Node.h"
#include "Nodes/DecisionNode.h"
#include "Nodes/LeafNode.h"
#include "../DTDataset.h"

class DecisionTree {
private:
    std::unique_ptr<Node> _root;
    std::vector<std::string> _headers;
    size_t _targetColumn = 0;
    std::ostringstream _buildingProcessOSS;

    void PrintPredictionsTable(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& predictions) const;

public:
    void SetRoot(std::unique_ptr<Node> root);
    void SetHeaders(const std::vector<std::string>& headers);
    void SetTargetColumn(size_t targetColumn);

    std::string Predict(const std::vector<std::string>& sample) const;
    void Predict(const std::vector<std::vector<std::string>>& testData) const;
    void Predict(const DTDataset& testDataset) const;

    std::ostringstream& GetBuildingProcessOSS();
    void ClearBuildingProcessOSS();
    std::string GetBuildingProcessDescr() const;
    void PrintTree() const;
};

