cmake_minimum_required(VERSION 3.18)
project(optlib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(optlib_core STATIC
  cpp/src/black_scholes.cpp
  cpp/src/monte_carlo.cpp
)
target_include_directories(optlib_core PUBLIC cpp/include)
target_compile_options(optlib_core PRIVATE -Wall -Wextra -Wpedantic)

# ---- C++ unit tests (CTest) -------------------------------------------------
enable_testing()
file(GLOB OPTLIB_CPP_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_*.cpp)
foreach(src ${OPTLIB_CPP_TESTS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE optlib_core)
  target_include_directories(${name} PRIVATE tests/cpp)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
