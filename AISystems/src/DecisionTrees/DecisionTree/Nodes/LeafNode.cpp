#include <../include/DecisionTrees/DecisionTree/Nodes/LeafNode.h>

std::string LeafNode::Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const {
    return _result;
}

void LeafNode::Print(int depth, bool isLastChild, const std::string& parentIndent) const {
    std::string currentIndent = parentIndent + (isLastChild ? "    " : "|   ");
    std::cout << currentIndent << "`-- Решение: \"" << "\033[1;32m\033[4m" << _result << "\033[0m\""
        << "\n" << currentIndent << "\n";
}