cmake_minimum_required(VERSION 4.3)
project(hw2 LANGUAGES CUDA)

set_source_files_properties(q1.cpp q2.cpp q3.cpp PROPERTIES LANGUAGE CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(hw2 main.cu utils.h)
add_executable(q1 q1.cpp utils.h)
add_executable(q2 q2.cpp utils.h)
add_executable(q3 q3.cpp utils.h)

set_target_properties(hw2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
