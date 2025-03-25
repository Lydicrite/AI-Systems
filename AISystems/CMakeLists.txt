cmake_minimum_required(VERSION 3.20)
project(AlSystems VERSION 0.1 LANGUAGES CXX)

# Настройки стандарта C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Директория для сборки
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

add_library( 
    AlSystemsCore STATIC
    "src/DecisionTrees/DTDataset.cpp"
    "src/DecisionTrees/DecisionTree/DecisionTree.cpp"
    "src/DecisionTrees/DecisionTree/Nodes/DecisionNode.cpp" 
    "src/DecisionTrees/DecisionTree/Nodes/LeafNode.cpp"
    "src/DecisionTrees/BuildAlgorithms/ID3.cpp"
    "src/Utils/ConsoleColor.cpp"

    "include/DecisionTrees/DTDataset.h"
    "include/DecisionTrees/DecisionTree/Nodes/DecisionNode.h" 
    "include/DecisionTrees/DecisionTree/Nodes/LeafNode.h" 
    "include/DecisionTrees/DecisionTree/Nodes/Node.h"
    "include/DecisionTrees/BuildAlgorithms/ID3.h"
    "src/Utils/ConsoleColor.cpp" 
    "include/Utils/ConsoleColor.h")

# Добавьте источник в исполняемый файл этого проекта.
add_executable (AISystems 
"src/AISystemsMain.cpp" "include/AISystemsMain.h" 
"src/DecisionTrees/DecisionTree/Nodes/DecisionNode.cpp" 
"src/DecisionTrees/DecisionTree/Nodes/LeafNode.cpp"
"src/DecisionTrees/BuildAlgorithms/ID3.cpp" 

"include/DecisionTrees/DecisionTree/Nodes/DecisionNode.h" 
"include/DecisionTrees/DecisionTree/Nodes/LeafNode.h" 
"include/DecisionTrees/DecisionTree/Nodes/Node.h" 
"include/DecisionTrees/BuildAlgorithms/ID3.h"
"include/Utils/ConsoleColor.h")


if (CMAKE_VERSION VERSION_GREATER 3.20)
  set_property(TARGET AISystems PROPERTY CXX_STANDARD 20)
endif()

# Подключение заголовков
target_include_directories(AlSystemsCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Линковка компонентов
target_link_libraries(AISystems PRIVATE AlSystemsCore)

# Копирование папки datasets в директорию с исполняемым файлом
install(DIRECTORY datasets/ DESTINATION datasets)



# Копирование датасетов
add_custom_target(CopyData ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/datasets"
    COMMAND ${CMAKE_COMMAND} -E copy 
        "${CMAKE_BINARY_DIR}/datasets/"
        "${CMAKE_SOURCE_DIR}/datasets/weather_data.csv" 
    DEPENDS AlSystems
    COMMENT "Копирование данных"
)