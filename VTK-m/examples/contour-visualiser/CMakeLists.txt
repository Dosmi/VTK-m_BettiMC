cmake_minimum_required(VERSION 3.12)
project(ContourVisualiser LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find VTK-m package (CONFIG mode)
find_package(VTKm REQUIRED QUIET)

# Check again for VTKm
if(NOT VTKm_FOUND)
    message(FATAL_ERROR "VTKm not found")
endif()

if(NOT TARGET vtkm::io OR NOT TARGET vtkm::filter_scalar_topology OR NOT TARGET vtkm::filter_contour)
  message(FATAL_ERROR "VTKm libraries required (vtkm::io, vtkm::filter_scalar_topology, vtkm::filter_contour) are missing.")
  return()
endif()

# Source files
set(SOURCES
    ContourVisualiser.cpp
    ct.cpp
    filter.cpp
    interface.cpp
    mc.cpp
    triangle.cpp
    jonas/CinemaExporter.cpp
    jonas/VTKDataSetWriter2.cpp
)

# Define the executable
add_executable(ContourVisualiser ${SOURCES})

# Link VTKm with official imported targets
target_link_libraries(ContourVisualiser PRIVATE
    vtkm::filter_scalar_topology
    vtkm::filter_contour
    vtkm::cont
    vtkm::worklet
    vtkm::filter
    vtkm::rendering
    vtkm::io
)

# Add other necessary includes explicitly if required
target_include_directories(ContourVisualiser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/jonas
    ${CMAKE_CURRENT_SOURCE_DIR}/libraries
)

# Recommended additional compile warnings
target_compile_options(ContourVisualiser PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
