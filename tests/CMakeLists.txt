set (BOOST_USE_STATIC_LIBS OFF)
find_package (Boost COMPONENTS unit_test_framework REQUIRED)
add_definitions (-DBOOST_TEST_DYN_LINK)

include_directories (${TEST_SOURCE_DIR}/include
                     ${Boost_INCLUDE_DIRS}
                     )

add_executable (qlearn_test qlearn.cpp)

target_link_libraries(qlearn_test PUBLIC ${Boost_LIBRARIES} prlearnStatic glpk)

add_test(NAME qlearn_test COMMAND qlearn_test)

