#pragma once
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <unordered_map>

class DTDataset
{
private:
    std::vector<std::string> _headers;
    std::vector<std::vector<std::string>> _data;
    size_t _numColumns = 0;
    bool _headerLoaded = false;
    size_t _targetColumn = 0;
    double _targetEntropy = 0;

    std::vector<std::string> Split(const std::string& line, char delimiter);
    void ValidateRow(const std::vector<std::string>& row, size_t lineIndex) const;
    std::vector<size_t> CalculateColumnWidths() const;

public:
    void LoadFromFile(const std::string& filename, char delimiter, bool hasHeader);

    const std::vector<std::string>& GetHeaders() const;
    const std::vector<std::vector<std::string>>& GetData() const;
    size_t RowCount() const;
    size_t ColumnCount() const;
    size_t GetColumnIndex(const std::string& columnName) const;
    std::string GetColumnHeader(size_t columnIndex) const;

    void PrintSummary(size_t previewRows) const;
    void PrintDataSlice(size_t previewRows) const;
    void PrintColumnStats(size_t columnIndex) const;
    void PrintColumnStats(const std::string& columnName) const;
    void PrintDataStats() const;

    void SortByColumn
    (
        size_t columnIndex,
        const std::function<bool(const std::string&, const std::string&)>& comparator
    );
    void SortByColumn(
        const std::string& columnName,
        const std::function<bool(const std::string&, const std::string&)>& comparator
    );

    std::unordered_set<std::string> GetUniqueValues(size_t columnIndex) const;
    std::unordered_set<std::string> GetUniqueValues(const std::string& columnName) const;

    void SetTargetColumn(const std::string& columnName);
    void SetTargetColumn(size_t columnIndex);
    size_t GetTargetColumn() const;
    std::string GetTargetColumnHeader() const;

    std::unordered_map<std::string, size_t> GetClassDistribution() const;
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>>
        GetClassDistributionForFeature(size_t featureIndex) const;

    double CalculateEntropy() const;
    double GetTargetEntropy() const;

    DTDataset GetFeatureValueSubset(size_t featureColumn, const std::string& value) const;
    DTDataset GetSubsetWithoutColumn(size_t columnIndex) const;
    DTDataset GetSubsetWithoutColumn(const std::string& columnName) const;
    DTDataset GetSubsetWithoutRow(size_t rowIndex) const;
    DTDataset GetSubsetWithoutRows(size_t startIndex, size_t endIndex) const;
};