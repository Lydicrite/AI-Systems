#include <../include/DecisionTrees/DecisionTree/DecisionTree.h>

void DecisionTree::PrintPredictionsTable(const std::vector<std::vector<std::string>>& data, const std::vector<std::string>& predictions) const {
    if (data.empty()) {
        std::cout << "Нет данных для отображения" << std::endl;
        return;
    }

    std::vector<std::string> tableHeaders = _headers;
    if (!tableHeaders.empty() && _targetColumn < tableHeaders.size()) {
        tableHeaders.erase(tableHeaders.begin() + _targetColumn);
    }

    std::vector<std::vector<std::string>> filteredData;
    for (const auto& row : data) {
        std::vector<std::string> filteredRow = row;
        if (_targetColumn < filteredRow.size()) {
            filteredRow.erase(filteredRow.begin() + _targetColumn);
        }
        filteredData.push_back(filteredRow);
    }

    tableHeaders.emplace_back("Предсказание");

    std::vector<size_t> columnWidths;
    for (size_t i = 0; i < tableHeaders.size(); ++i) {
        size_t maxWidth = tableHeaders[i].size();
        for (const auto& row : filteredData) {
            if (i < row.size()) {
                maxWidth = std::max(maxWidth, row[i].size());
            }
        }
        if (i == tableHeaders.size() - 1) { // Для колонки Prediction
            for (const auto& pred : predictions) {
                maxWidth = std::max(maxWidth, pred.size());
            }
        }
        columnWidths.push_back(maxWidth + 2);
    }

    for (size_t i = 0; i < tableHeaders.size(); ++i) {
        std::cout << std::left << std::setw(columnWidths[i]) << tableHeaders[i] << "|";
    }
    std::cout << "\n";

    for (size_t width : columnWidths) {
        std::cout << std::string(width, '-') << "+";
    }
    std::cout << "\n";

    for (size_t i = 0; i < filteredData.size(); ++i) {
        const auto& row = filteredData[i];
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::left << std::setw(columnWidths[j]) << row[j] << "|";
        }
        std::cout << std::left << std::setw(columnWidths.back()) << predictions[i] << "|\n";
    }
}



void DecisionTree::SetRoot(std::unique_ptr<Node> root) {
    _root = std::move(root);
}

void DecisionTree::SetHeaders(const std::vector<std::string>& headers) {
    _headers = headers;
}

void DecisionTree::SetTargetColumn(size_t targetColumn) {
    _targetColumn = targetColumn;
}



std::string DecisionTree::Predict(const std::vector<std::string>& sample) const {
    if (!_root)
        throw std::logic_error("Дерево не обучено");

    // Проверка соответствия количества признаков

    if (sample.size() != _headers.size() - 1) {
        std::stringstream ss;
        ss << "Несоответствие количества признаков. Ожидалось " << _headers.size() - 1
            << ", получено " << sample.size();
        throw std::invalid_argument(ss.str());
    }

    return _root->Predict(sample, _headers);
}

void DecisionTree::Predict(const std::vector<std::vector<std::string>>& testData) const {
    if (!_root) {
        throw std::logic_error("Дерево не обучено");
    }

    // Проверка соответствия количества признаков
    for (const auto& sample : testData) {
        if (sample.size() != _headers.size() - 1) {
            std::stringstream ss;
            ss << "Несоответствие количества признаков. Ожидалось " << _headers.size() - 1
                << ", получено " << sample.size();
            throw std::invalid_argument(ss.str());
        }
    }

    // Сбор предсказаний
    std::vector<std::string> predictions;
    for (const auto& sample : testData) {
        predictions.push_back(_root->Predict(sample, _headers));
    }

    // Вывод таблицы
    PrintPredictionsTable(testData, predictions);
}

void DecisionTree::Predict(const DTDataset& testDataset) const {
    if (!_root) {
        throw std::logic_error("Дерево не обучено");
    }

    // Проверка количества столбцов
    if (testDataset.ColumnCount() != _headers.size()) {
        std::stringstream ss;
        ss << "Несоответствие количества признаков. Ожидалось " << _headers.size()
            << ", получено " << testDataset.ColumnCount();
        throw std::invalid_argument(ss.str());
    }

    // Проверка заголовков (если есть)
    if (testDataset.GetHeaders().size() > 0 && testDataset.GetHeaders() != _headers) {
        throw std::invalid_argument("Заголовки тестовых данных не совпадают с ожидаемыми");
    }

    // Сбор данных и предсказаний
    auto testData = testDataset.GetSubsetWithoutColumn(testDataset.GetTargetColumn()).GetData();

    std::vector<std::string> predictions;
    for (const auto& row : testData) {
        predictions.push_back(_root->Predict(row, _headers));
    }

    // Вывод таблицы
    PrintPredictionsTable(testData, predictions);
}



std::ostringstream& DecisionTree::GetBuildingProcessOSS() {
    return _buildingProcessOSS;
}

void DecisionTree::ClearBuildingProcessOSS() {
    _buildingProcessOSS.str("");
    _buildingProcessOSS.clear();
}

std::string DecisionTree::GetBuildingProcessDescr() const {
    return _buildingProcessOSS.str();
}

void DecisionTree::PrintTree() const {
    if (_root)
        _root->Print(0, false, "");
    else
        std::cout << "Дерево пустое\n";
}