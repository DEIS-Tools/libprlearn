cmake_minimum_required(VERSION 3.12)
project(PRLearn C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_library(prlearn ${HEADER_FILES} MLearning.cpp SimpleMLearning.cpp RefinementTree.cpp structs.cpp)
target_include_directories (prlearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
