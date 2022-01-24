cmake_minimum_required(VERSION 2.4)

project(Tarea_01)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
include_directories(Modules/)
add_executable(Main.out Modules/files.cpp Modules/nodes.cpp Modules/lines.cpp main.cpp)
