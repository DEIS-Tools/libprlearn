cmake_minimum_required(VERSION 3.7)
project(PRLearn C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Boost 1.54 REQUIRED COMPONENTS headers REQUIRED)

add_library(prlearn SHARED ${HEADER_FILES} MLearning.cpp SimpleMLearning.cpp RefinementTree.cpp structs.cpp)
add_library(prlearnStatic STATIC ${HEADER_FILES} MLearning.cpp SimpleMLearning.cpp RefinementTree.cpp structs.cpp)

target_include_directories(prlearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
target_include_directories(prlearnStatic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})
set_target_properties(prlearnStatic PROPERTIES OUTPUT_NAME prlearn)


install(TARGETS prlearn
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
install (FILES  MLearning.h
		propts.h
		QLearning.h
		RefinementTree.h
		SimpleMLearning.h
		SimpleRegressor.h
		structs.h
	DESTINATION include/prlearn)
