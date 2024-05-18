# CMakeList.txt : CMake project for VenomEngine, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project (imgui)

# Set source files for the library
file(GLOB SOURCES "*.c" "*.cpp" "*.h" "*.hpp")

add_library(imgui SHARED ${SOURCES})

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(imgui
    PROPERTIES FOLDER "ExternalLibraries"
)

target_compile_definitions(imgui PRIVATE IMGUI_API_EXPORTS)
