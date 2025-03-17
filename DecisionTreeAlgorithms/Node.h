#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <iomanip>
#include <iostream>

class Node {
public:
    virtual ~Node() = default;
    virtual std::string Predict(const std::vector<std::string>& sample, const std::vector<std::string>& headers) const = 0;
    virtual void Print(int depth, bool isLastChild, const std::string& parentIndent) const = 0;
};

