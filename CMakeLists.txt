cmake_minimum_required(VERSION 3.9)

set(ROUTINE sparseTest)

project("${ROUTINE}"
        LANGUAGES CUDA CXX)

add_executable(${ROUTINE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -O3")


target_sources(${ROUTINE}
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}.cc
)

target_include_directories(${ROUTINE}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${ROUTINE}
    PUBLIC cudart cusparse
)
