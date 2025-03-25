#pragma once
#include "Node.h"

class LeafNode : public Node {
private:
    std::string _result;

public:
    LeafNode(const std::string& result)
        : _result(result) {
    }

    std::string Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const override;
    void Print(int depth, bool isLastChild, const std::string& parentIndent) const override;
};