cmake_minimum_required(VERSION 3.7)

cmake_policy(SET CMP0048 NEW)

project(libprlearn VERSION 1.0.0 LANGUAGES CXX)
set(PRLEARN_NAME "libprlearn")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -Wall -Wextra -Wpedantic")
option(PRLEARN_TEST "Build unit tests" OFF)
option(PRLEARN_GetDependencies, "Get external dependencies" ON)

set(PRLEARN_TARGETDIR "${CMAKE_BINARY_DIR}/${VERIFYPN_NAME}" CACHE PATH "Traget directory for build files")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PRLEARN_TARGETDIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PRLEARN_TARGETDIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PRLEARN_TARGETDIR}/bin)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external CACHE PATH "Install location for external dependencies")

if (PRLEARN_GetDependencies)
    include(ExternalProject)
    if (WIN32) #If windows 32 or 64
      set(GLPK_CFLAGS "-D __WOE__ -O3" )
    else(WIN32)
      set(GLPK_CFLAGS "-O3" )
    endif(WIN32)
    ExternalProject_add(glpk-ext
        URL https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
        URL_HASH SHA512=997e8e599ff1718a08c66b86eadd0e01f4644899f1e95920f8ae91d66b4d8361021766b346845f4dcbcfe667b41ab72ea3d377017a0ebf85d7ece091cfd81375
	CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} <SOURCE_DIR>/configure --enable-shared=no --prefix=${EXTERNAL_INSTALL_LOCATION} CFLAGS=${GLPK_CFLAGS}
	BUILD_COMMAND make
        INSTALL_COMMAND make install
    )
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
    link_directories(${EXTERNAL_INSTALL_LOCATION}/lib64)
    include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
endif()



#actual library
add_subdirectory(src)


if (PRLEARN_TEST)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()