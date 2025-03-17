#pragma once
#include "Node.h"

class DecisionNode : public Node {
private:
    std::string _index;
    std::string _featureName;
    std::unordered_map<std::string, std::unique_ptr<Node>> _children;

public:
    DecisionNode(const std::string& featureName, const std::string& index)
        : _featureName(featureName), _index(index) {
    }

    void AddChild(const std::string& value, std::unique_ptr<Node> child);
    std::string Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const override;
    void Print(int depth, bool isLastChild, const std::string& parentIndent) const override;
};