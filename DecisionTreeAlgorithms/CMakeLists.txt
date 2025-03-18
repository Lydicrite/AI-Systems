cmake_minimum_required(VERSION 3.12)
project(AI_Systems_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES
    "DTBuildAlgorithms/*.cpp"
    "DecisionTree/*.cpp"
    "main.cpp"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/DecisionTree
    ${CMAKE_CURRENT_SOURCE_DIR}/DTBuildAlgorithms
)

# Создаем исполняемый файл
add_executable(AI_Systems_Project ${SOURCES})