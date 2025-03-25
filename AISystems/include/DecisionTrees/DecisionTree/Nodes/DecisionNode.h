#pragma once
#include "Node.h"

class DecisionNode : public Node {
private:
    std::string _featureName;
    std::unordered_map<std::string, std::unique_ptr<Node>> _children;

public:
    DecisionNode(const std::string& featureName)
        : _featureName(featureName) {
    }

    void AddChild(const std::string& value, std::unique_ptr<Node> child);
    std::string Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const override;
    void Print(int depth, bool isLastChild, const std::string& parentIndent) const override;
};