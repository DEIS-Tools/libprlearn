cmake_minimum_required(VERSION 3.10)

cmake_policy(SET CMP0048 NEW)

project(libprlearn VERSION 1.0.0 LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -Wall -Wpedantic")
set(CMAKE_CXX_STANDARD 17)

#actual library
add_subdirectory(src)
