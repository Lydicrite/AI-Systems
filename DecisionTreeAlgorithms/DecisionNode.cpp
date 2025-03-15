#include "DecisionNode.h"

void DecisionNode::AddChild(const std::string& value, std::unique_ptr<Node> child) {
    _children[value] = std::move(child);
}

std::string DecisionNode::Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const {
    auto it = std::find(headers.begin(), headers.end(), _featureName);
    if (it == headers.end()) return "(неизвестно)";
    size_t featureIndex = it - headers.begin();

    if (featureIndex >= sample.size()) 
        return "(неизвестно)";

    auto childIt = _children.find(sample[featureIndex]);
    if (childIt == _children.end()) 
        return "(неизвестно)";

    return childIt->second->Predict(sample, headers);
}

void DecisionNode::Print(int depth = 0) const {
    std::string indent(depth * 4, ' ');

    // Вывод текущего узла
    std::cout << indent << "[" << _index << "] -> " << "признак \"" << "\033[1;36m" << _featureName << "\033[0m\"" << "\n";

    // Обработка дочерних элементов
    size_t childCount = 0;
    const size_t totalChildren = _children.size();
    for (const auto& pair : _children) {
        std::cout << indent << "[" << _index << "] -> " << "значение: \"" << "\033[1;31m" << pair.first << "\033[0m\"" << "\n";
        pair.second->Print(depth + 1);
        std::cout << '\n';
    }
}