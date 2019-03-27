cmake_minimum_required(VERSION 3.1)

file (GLOB TESTBDLA_SOURCE  "testmain.c")
source_group("bdla_test_source" FILES ${TESTBDLA_SOURCE})


# And add the main as an executable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
enable_testing()
add_executable(bdla_tests ${TESTBDLA_SOURCE})
target_link_libraries(bdla_tests PUBLIC bdla)
target_compile_definitions(bdla_tests PRIVATE _CRT_SECURE_NO_WARNINGS)


