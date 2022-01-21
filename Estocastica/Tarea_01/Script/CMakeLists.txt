cmake_minimum_required(VERSION 2.4)

project(Tarea_01)

set(CMAKE_CXX_STANDARD 17)
include_directories(Modules/)
add_executable(Main.out Modules/files.cpp Modules/nodes.cpp Modules/data.cpp main.cpp)
