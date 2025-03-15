#pragma once
#include "Node.h"
#include "DecisionNode.h"
#include "LeafNode.h"

class DecisionTree {
private:
    std::unique_ptr<Node> _root;
    std::vector<std::string> _headers;

public:
    void SetRoot(std::unique_ptr<Node> root) {
        _root = std::move(root);
    }

    void SetHeaders(const std::vector<std::string>& headers) {
        _headers = headers;
    }

    std::string Predict(const std::vector<std::string>& sample) const {
        if (!_root) 
            throw std::logic_error("Дерево не обучено");

        return _root->Predict(sample, _headers);
    }

    void PrintTree() const {
        if (_root) 
            _root->Print(0);
        else 
            std::cout << "Дерево пустое\n";
    }
};

