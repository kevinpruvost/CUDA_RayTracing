# CMakeList.txt : CMake project for VenomEngine, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project (glad)

# Set source files for the library
file(GLOB SOURCES "src/*.c" "include/glad/*.h" "include/KHR/*.h")

add_library(glad STATIC ${SOURCES})

target_include_directories(glad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(glad
    PROPERTIES FOLDER "ExternalLibraries"
)
