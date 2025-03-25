#include <../include/DecisionTrees/DecisionTree/Nodes/DecisionNode.h>

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

void DecisionNode::Print(int depth, bool isLastChild, const std::string& parentIndent) const {
    std::string currentIndent;

    // Формируем отступ для текущего узла
    if (depth > 0) {
        currentIndent = parentIndent + (isLastChild ? "    " : "|   ");
    }

    // Вывод текущего узла
    std::cout << currentIndent << (depth == 0 ? "|-- " : "|-- ")
        << "Признак: \"" << "\033[1;36m" << _featureName << "\033[0m\"\n";

    // Обработка дочерних элементов
    size_t childIndex = 0;
    for (const auto& pair : _children) {
        bool isLast = (childIndex == _children.size() - 1);
        std::string childConnector = isLast ? "`-- " : "|-- ";

        // Вывод значения
        std::cout << currentIndent << (isLast ? "    " : "|   ")
            << childConnector << "Значение: \"" << "\033[1;31m" << pair.first << "\033[0m\"" << "\n";

        // Рекурсивный вызов для дочернего узла
        pair.second->Print(depth + 1, isLast, currentIndent + (isLast ? "    " : "|   "));
        childIndex++;
    }
}