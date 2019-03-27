cmake_minimum_required(VERSION 3.0)
project(bdla C)

option(BUILD_UNIT_TESTS "Builds tests" OFF)
option(USE_OPENMP "Use the OpenMP multithreading" ON)
option(BUILD_STATIC_LIBRARY "Builds static library instead of shared" OFF)

# Everything is placed in the one dictionary. Life is easier.
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif(USE_OPENMP)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
file (GLOB BDLA_INCLUDE "include/bdla/libbdla.h") # So shoot me for GLOBing.
file (GLOB BDLA_SOURCE  "src/*.[ch]")
source_group("" FILES ${bdla})
source_group("include" FILES ${BDLA_INCLUDE})
source_group("source" FILES ${BDLA_SOURCE})

include_directories (include/bdla)
if(BUILD_STATIC_LIBRARY)
	add_library(bdla ${BDLA_INCLUDE} ${BDLA_SOURCE})
	target_compile_definitions(bdla PRIVATE BDLA_EXPORT=)
else()
	add_library(bdla SHARED ${BDLA_INCLUDE} ${BDLA_SOURCE})
	if(WIN32)
		target_compile_definitions(bdla PRIVATE BDLA_EXPORT=__declspec\(dllexport\))
	else()
		target_compile_definitions(bdla PRIVATE BDLA_EXPORT=)
	endif()
endif(BUILD_STATIC_LIBRARY)
# We don't want warnings...
target_compile_definitions(bdla PRIVATE _CRT_SECURE_NO_WARNINGS)

#find_package(bsv CONFIG REQUIRED)
#target_link_libraries(bdla PUBLIC bsv)
#target_include_directories(bdla PUBLIC bsv)

find_package(OpenBLAS CONFIG REQUIRED)
target_link_libraries(bdla PRIVATE OpenBLAS_LIBRARIES)
target_include_directories(bdla PRIVATE OpenBLAS_INCLUDE_DIRS)

if (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    link_libraries(bdla m)   # Maths std library.
endif()
						
set_property(TARGET bdla PROPERTY FOLDER "libraries")
set_target_properties(bdla PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
                      PUBLIC_HEADER "${BDLA_INCLUDE}")

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif(BUILD_UNIT_TESTS)

# Exporting targets
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bdla
    EXPORT bdla-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT bdla-targets
  FILE
    bdla-config.cmake
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/bdla
)
