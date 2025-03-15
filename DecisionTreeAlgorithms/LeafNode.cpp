#include "LeafNode.h"

std::string LeafNode::Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const {
    return _result;
}

void LeafNode::Print(int depth = 0) const {
    std::string indent(depth * 4, ' ');
    std::cout << indent << "[" << _index << "] -> " << "Решение: \"" << "\033[1;32m\033[4m" << _result << "\033[0m\"" << "\n";
}