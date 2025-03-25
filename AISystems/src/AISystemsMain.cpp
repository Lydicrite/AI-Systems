#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include <../include/Utils/ConsoleColor.h>
#include <../include/AISystemsMain.h>
#include <../include/DecisionTrees/DTDataset.h>
#include <../include/DecisionTrees/BuildAlgorithms/ID3.h>

int wmain(int argc, wchar_t* argv[], wchar_t* envp[]) {
    if (ConsoleColor::isTerminal()) {
        ConsoleColor::enableColorSupport();
    }
    
    setlocale(LC_ALL, "ru_RU");

    try {
        // Создание набора данных из CSV-файла
        DTDataset dataset;

        std::string dataset_path = "datasets\\weather_data.csv";
        std::ifstream test_file(dataset_path);
        if (!test_file) {
            throw std::runtime_error("Не найден файл: " + dataset_path);
        }
        dataset.LoadFromFile(dataset_path, ';', true);

        dataset.PrintSummary(0);
        dataset.SetTargetColumn("Play");
        std::cout << "\nЭнтропия признака \"Play\": " << dataset.CalculateEntropy() << '\n';

        std::cout << "\n\n-------------------- Построение дерева решений с помощью ID3 и работа с ним --------------------\n\n";
        {
            // Построение дерева по ID3
            DecisionTree tree = ID3::Train(dataset);

            // Вывод построения по ID3
            std::cout << tree.GetBuildingProcessDescr();

            // Тест "предсказания"
            std::vector<std::string> sample = { "Sunny", "Cool", "Normal", "Weak" };
            auto prediction = tree.Predict(sample);
            std::cout << "Предсказания целевого признака \"Play\" "
                << "по данным {\"Sunny\", \"Cool\", \"Normal\", \"Weak\"}: "
                << "\"" << prediction << "\"\n\n";

            // Тест всех "предсказания"
            std::cout << "Предсказания целевого признака \"Play\" по всему исходному набору данных: \n";
            auto testDataset = dataset;
            tree.Predict(testDataset);

            // Вывод структуры
            std::cout << "\n\n\nСтруктура дерева:\n\n";
            tree.PrintTree();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "\033[31mОшибка:\033[0m " << e.what() << std::endl;
    }

    std::cout << "\n\n\n";
    system("pause");

    return 0;
}

/*
        std::cout << "\n\n\n";

        dataset.SortByColumn
        (
            "Outlook",

            [](const std::string& a, const std::string& b) {
            return std::lexicographical_compare(
                a.begin(), a.end(),
                b.begin(), b.end(),
                [](char c1, char c2) {
                    return std::tolower(c1) < std::tolower(c2);
                });
            }
        );

        dataset.PrintDataSlice(0);
        */
