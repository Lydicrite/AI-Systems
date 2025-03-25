#include <../include/DecisionTrees/DTDataset.h>

std::vector<std::string> DTDataset::Split(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(line);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty() && token.front() == '\"') {
            token = std::string(token.begin() + 1, token.end());
        }
        if (!token.empty() && token.back() == '\"') {
            token = std::string(token.begin(), token.end() - 1);
        }
        tokens.push_back(token);
    }
    return tokens;
}

void DTDataset::ValidateRow(const std::vector<std::string>& row, size_t lineIndex) const {
    if (row.size() != _numColumns) {
        std::stringstream ss;
        ss << "������ � ������ " << lineIndex
            << ": ��������� " << _numColumns
            << " ��������, �������� " << row.size();
        throw std::invalid_argument(ss.str());
    }

    for (size_t i = 0; i < row.size(); ++i) {
        if (row[i].empty()) {
            std::stringstream ss;
            ss << "������ �������� � ������ " << lineIndex
                << ", ������� " << (_headerLoaded ? _headers[i] : std::to_string(i));
            throw std::invalid_argument(ss.str());
        }
    }
}

std::vector<size_t> DTDataset::CalculateColumnWidths() const {
    std::vector<size_t> widths(_numColumns, 0);

    if (_headerLoaded) {
        for (size_t i = 0; i < _numColumns; ++i) {
            widths[i] = _headers[i].size();
        }
    }

    for (const auto& row : _data) {
        for (size_t i = 0; i < _numColumns; ++i) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }

    for (auto& w : widths) w += 2;

    return widths;
}



void DTDataset::LoadFromFile(const std::string& filename, char delimiter = ',', bool hasHeader = true) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("���� �� ������: " + filename);
    }

    _data.clear();
    _headers.clear();
    _numColumns = 0;
    _headerLoaded = false;

    std::string line;
    size_t lineNumber = 0;

    if (hasHeader) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("���� ����, �� �������� ���������");
        }
        lineNumber++;
        _headers = Split(line, delimiter);
        _numColumns = _headers.size();
        _headerLoaded = true;

        if (_headers.empty()) {
            throw std::invalid_argument("��������� �� �������� ������");
        }
    }

    while (std::getline(file, line)) {
        lineNumber++;
        if (line.empty()) continue;

        auto row = Split(line, delimiter);

        if (_numColumns == 0) {
            _numColumns = row.size();
            if (_numColumns == 0) {
                throw std::invalid_argument("������ ������ ������ �����");
            }
        }

        ValidateRow(row, lineNumber);
        _data.push_back(row);
    }

    if (_data.empty()) {
        throw std::runtime_error("���� �� �������� ������");
    }

    _targetColumn = _data[0].size() - 1;
}

const std::vector<std::string>& DTDataset::GetHeaders() const {
    return _headers;
}

const std::vector<std::vector<std::string>>& DTDataset::GetData() const {
    return _data;
}

size_t DTDataset::RowCount() const {
    return _data.size();
}

size_t DTDataset::ColumnCount() const {
    return _numColumns;
}

size_t DTDataset::GetColumnIndex(const std::string& columnName) const {
    if (!_headerLoaded) {
        throw std::logic_error("��������� �� ���������");
    }

    auto it = std::find(_headers.begin(), _headers.end(), columnName);
    if (it == _headers.end()) {
        std::stringstream ss;
        ss << "������� '" << columnName << "' �� ������. ��������� �������: ";
        for (size_t i = 0; i < _headers.size(); ++i) {
            ss << "\n  " << i << ") " << _headers[i];
        }
        throw std::invalid_argument(ss.str());
    }

    return static_cast<size_t>(it - _headers.begin());
}

std::string DTDataset::GetColumnHeader(size_t columnIndex) const {
    if (columnIndex >= _numColumns) {
        std::stringstream ss;
        ss << "������������ ������ �������: " << columnIndex
            << " (��������� 0-" << (_numColumns - 1) << ")";
        throw std::out_of_range(ss.str());
    }

    if (!_headerLoaded)
        return "Column " + columnIndex;

    return _headers[columnIndex];
}


void DTDataset::PrintSummary(size_t previewRows = 5) const {
    std::cout << "����� ���������� � ������ ������: "
        << "\n\t��������: " << _numColumns
        << "\n\t�����: " << _data.size() << "\n";

    if (_headerLoaded) {
        size_t count = 0;
        std::cout << "\n���������: ";
        for (const auto& h : _headers) {
            std::cout << '\"' << h << '\"';
            if (++count < _headers.size())
                std::cout << ", ";
            else
                std::cout << '.';
        }
        std::cout << "\n";
    }

    std::cout << '\n';
    PrintDataStats();

    PrintDataSlice(previewRows);
}

void DTDataset::PrintDataSlice(size_t previewRows = 5) const {
    const size_t rowsToShow = previewRows != 0 ?
        std::min(previewRows, _data.size()) :
        _data.size();

    if (rowsToShow == 0) {
        std::cout << "��� ������ ��� �����������\n";
        return;
    }

    std::cout << "\n������ (������ " << rowsToShow << " �����): \n";

    auto widths = CalculateColumnWidths();

    // ���������
    if (_headerLoaded) {
        for (size_t i = 0; i < _numColumns; ++i) {
            std::cout << std::left << std::setw(widths[i]) << _headers[i] << " |";
        }
        std::cout << "\n";

        // �������������� �����
        for (size_t i = 0; i < _numColumns; ++i) {
            std::cout << std::string(widths[i], '-') << "-+";
        }
        std::cout << "\n";
    }

    // ������
    for (size_t i = 0; i < rowsToShow; ++i) {
        for (size_t j = 0; j < _numColumns; ++j) {
            std::cout << std::left << std::setw(widths[j]) << _data[i][j] << " |";
        }
        std::cout << "\n";
    }
}

void DTDataset::PrintColumnStats(size_t columnIndex) const {
    if (columnIndex >= _numColumns) {
        std::stringstream ss;
        ss << "������������ ������ �������: " << columnIndex
            << " (��������� 0-" << (_numColumns - 1) << ")";
        throw std::out_of_range(ss.str());
    }

    auto unique = GetUniqueValues(columnIndex);
    std::cout << "���������� ��� ������� "
        << (_headerLoaded ? _headers[columnIndex] : std::to_string(columnIndex)) << ": \n"
        << "\t���������� ��������: " << unique.size() << "\n"
        << "\t��������: ";

    size_t count = 0;
    for (const auto& val : unique) {
        std::cout << '\"' << val << '\"';
        if (++count < unique.size())
            std::cout << ", ";
        else
            std::cout << '.';
    }
    std::cout << "\n";
}

void DTDataset::PrintColumnStats(const std::string& columnName) const {
    PrintColumnStats(GetColumnIndex(columnName));
}

void DTDataset::PrintDataStats() const {
    if (_numColumns == 0 || _data.empty()) {
        std::cout << "��� ������ ��� ����������� ����������\n";
        return;
    }

    // ������������ ������ ��� ������� � ����������
    size_t nameWidth = 0;
    if (_headerLoaded) {
        for (const auto& h : _headers) {
            nameWidth = std::max(nameWidth, h.size());
        }
    }
    else {
        nameWidth = std::to_string(_numColumns - 1).size() + 8;
    }
    nameWidth += 5;

    // �����
    std::cout << "���������� ������ �� ��������: \n";
    std::cout << std::string(nameWidth, '-') << "-|----------------------------------------\n";
    std::cout << std::left << std::setw(nameWidth) << "�������� �������"
        << " | ���������� ��������\n";
    std::cout << std::string(nameWidth, '-') << "-+----------------------------------------\n";

    // ��� ������� �������
    for (size_t i = 0; i < _numColumns; ++i) {
        // �������� �������
        std::string colName = _headerLoaded ? _headers[i] : ("Column " + std::to_string(i));
        std::cout << std::left << std::setw(nameWidth) << colName << " | ";

        // ���������� ��������
        auto unique = GetUniqueValues(i);
        std::cout << unique.size() << ": ";
        size_t count = 0;
        for (const auto& val : unique) {
            std::cout << "\"" << val << "\"";
            if (++count < unique.size()) std::cout << ", ";
        }

        std::cout << "\n";

        if (i == _numColumns - 1)
            std::cout << std::string(nameWidth, '-') << "-|----------------------------------------\n";
        else
            std::cout << std::string(nameWidth, '-') << "-+----------------------------------------\n";
    }
}



void DTDataset::SortByColumn
(
    size_t columnIndex,
    const std::function<bool(const std::string&, const std::string&)>& comparator = nullptr
)
{
    if (columnIndex >= _numColumns) {
        std::stringstream ss;
        ss << "������ ������� " << columnIndex << " ������� �� ������� [0, "
            << (_numColumns - 1) << "]";
        throw std::out_of_range(ss.str());
    }

    auto default_comparator = [](const std::string& a, const std::string& b) {
        return a < b;
        };

    auto& comp = comparator ? comparator : default_comparator;

    std::sort(_data.begin(), _data.end(),
        [columnIndex, &comp](const std::vector<std::string>& a,
            const std::vector<std::string>& b) {
                return comp(a[columnIndex], b[columnIndex]);
        });
}

void DTDataset::SortByColumn
(
    const std::string& columnName,
    const std::function<bool(const std::string&, const std::string&)>& comparator = nullptr
)
{
    SortByColumn(GetColumnIndex(columnName), comparator);
}



std::unordered_set<std::string> DTDataset::GetUniqueValues(size_t columnIndex) const {
    if (columnIndex >= _numColumns) {
        std::stringstream ss;
        ss << "������������ ������ �������: " << columnIndex
            << " (��������� 0-" << (_numColumns - 1) << ")";
        throw std::out_of_range(ss.str());
    }

    std::unordered_set<std::string> unique;
    for (const auto& row : _data) {
        unique.insert(row[columnIndex]);
    }
    return unique;
}

std::unordered_set<std::string> DTDataset::GetUniqueValues(const std::string& columnName) const {
    return GetUniqueValues(GetColumnIndex(columnName));
}



void DTDataset::SetTargetColumn(const std::string& columnName) {
    _targetColumn = GetColumnIndex(columnName);
}

void DTDataset::SetTargetColumn(size_t columnIndex) {
    if (columnIndex >= _numColumns)
        throw std::out_of_range("������������ ������ �������� �������");

    _targetColumn = columnIndex;
    _targetEntropy = CalculateEntropy();
}

size_t DTDataset::GetTargetColumn() const {
    return _targetColumn;
}

std::string DTDataset::GetTargetColumnHeader() const {
    if (!_headerLoaded)
        return "Column " + _targetColumn;

    return _headers[_targetColumn];
}



std::unordered_map<std::string, size_t> DTDataset::GetClassDistribution() const {
    std::unordered_map<std::string, size_t> dist;
    for (const auto& row : _data) {
        dist[row[_targetColumn]]++;
    }
    return dist;
}

std::unordered_map<std::string, std::unordered_map<std::string, size_t>>
DTDataset::GetClassDistributionForFeature(size_t featureIndex) const {
    if (featureIndex >= _numColumns) {
        throw std::out_of_range("������������ ������ ��������");
    }

    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> dist;
    for (const auto& row : _data) {
        const std::string& featureValue = row[featureIndex];
        const std::string& targetValue = row[_targetColumn];
        dist[featureValue][targetValue]++;
    }
    return dist;
}



double DTDataset::CalculateEntropy() const {
    auto dist = GetClassDistribution();
    double entropy = 0.0;
    size_t total = _data.size();
    if (total == 0)
        return 0.0;
    for (const auto& pair : dist) {
        double p = static_cast<double>(pair.second) / total;
        if (p > 0) entropy -= p * log2(p);
    }

    return entropy;
}

double DTDataset::GetTargetEntropy() const {
    return _targetEntropy;
}



DTDataset DTDataset::GetFeatureValueSubset(size_t featureColumn, const std::string& value) const {
    if (featureColumn >= _numColumns) {
        std::stringstream ss;
        ss << "������������ ������ �������: " << featureColumn
            << " (��������� 0-" << (_numColumns - 1) << ")";
        throw std::out_of_range(ss.str());
    }

    if (featureColumn == _targetColumn) {
        throw std::invalid_argument("������ ������� ������� �������");
    }

    DTDataset subset;
    subset._headers = _headers;
    subset._numColumns = _numColumns - 1;
    subset._headerLoaded = _headerLoaded;

    // ������ ������������, ������ ������� featureColumn
    for (const auto& row : _data) {
        if (row[featureColumn] == value) {
            std::vector<std::string> newRow = row;
            newRow.erase(newRow.begin() + featureColumn);
            subset._data.push_back(newRow);
        }
    }

    // ��������� ���������
    if (_headerLoaded) {
        subset._headers = _headers;
        subset._headers.erase(subset._headers.begin() + featureColumn);
    }

    if (subset._data.size() == 0) {
        std::stringstream ss;
        ss << "�� ���� ������� �� ������ ������� ������, ��� �������� ������� \""
            << _headers[featureColumn] << "\" ���� �� �������� \"" << value << "\"\n��������, "
            << "������� ������ ������ �������� ��� ��� ������� ��������";
        throw std::invalid_argument(ss.str());
    }

    return subset;
}

DTDataset DTDataset::GetSubsetWithoutColumn(size_t columnIndex) const {
    if (columnIndex >= _numColumns) {
        std::stringstream ss;
        ss << "������ ������� " << columnIndex << " ������� �� ������� [0, " << (_numColumns - 1) << "]";
        throw std::out_of_range(ss.str());
    }

    DTDataset subset;
    subset._headers = _headers;
    subset._numColumns = _numColumns - 1;
    subset._headerLoaded = _headerLoaded;

    // ������� ���������, ���� �� ����
    if (_headerLoaded) {
        subset._headers.erase(subset._headers.begin() + columnIndex);
    }

    // �������� ������ ��� ���������� �������
    for (const auto& row : _data) {
        std::vector<std::string> newRow = row;
        newRow.erase(newRow.begin() + columnIndex);
        subset._data.push_back(newRow);
    }

    if (subset._data.empty()) {
        throw std::runtime_error("�������������� ����� ������ ����");
    }

    return subset;
}

DTDataset DTDataset::GetSubsetWithoutColumn(const std::string& columnName) const {
    return GetSubsetWithoutColumn(GetColumnIndex(columnName));
}

DTDataset DTDataset::GetSubsetWithoutRow(size_t rowIndex) const {
    if (rowIndex >= _data.size()) {
        std::stringstream ss;
        ss << "������ ������ " << rowIndex << " ������� �� ������� [0, " << (_data.size() - 1) << "]";
        throw std::out_of_range(ss.str());
    }

    DTDataset subset = *this;
    subset._data.erase(subset._data.begin() + rowIndex);

    if (subset._data.empty()) {
        throw std::runtime_error("�������������� ����� ������ ����");
    }

    return subset;
}

DTDataset DTDataset::GetSubsetWithoutRows(size_t startIndex, size_t endIndex) const {
    if (startIndex > endIndex || endIndex >= _data.size()) {
        std::stringstream ss;
        ss << "������������ �������� [" << startIndex << ", " << endIndex
            << "]. ��������� [0, " << (_data.size() - 1) << "]";
        throw std::out_of_range(ss.str());
    }

    DTDataset subset = *this;
    subset._data.erase(
        subset._data.begin() + startIndex,
        subset._data.begin() + endIndex + 1
    );

    if (subset._data.empty()) {
        throw std::runtime_error("�������������� ����� ������ ����");
    }

    return subset;
}