#pragma once
#include "Node.h"

class LeafNode : public Node {
private:
    std::string _index;
    std::string _result;

public:
    LeafNode(const std::string& result, const std::string& index) 
        : _result(result), _index(index) {}

    std::string Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const override;
    void Print(int depth) const override;
};
