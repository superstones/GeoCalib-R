cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

project(v2x_gtsam_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GTSAM REQUIRED)
find_package(Pangolin REQUIRED)
find_package(nlohmann_json CONFIG QUIET)

add_executable(v2x_gtsam_viewer src/main.cpp)

if(DEFINED ENV{GEOCALIB_EXTRA_INCLUDE_DIR})
  target_include_directories(v2x_gtsam_viewer PRIVATE "$ENV{GEOCALIB_EXTRA_INCLUDE_DIR}")
endif()

target_link_libraries(v2x_gtsam_viewer PRIVATE gtsam)

if(TARGET Pangolin::Pangolin)
  target_link_libraries(v2x_gtsam_viewer PRIVATE Pangolin::Pangolin)
else()
  target_link_libraries(v2x_gtsam_viewer PRIVATE ${Pangolin_LIBRARIES})
  target_include_directories(v2x_gtsam_viewer PRIVATE ${Pangolin_INCLUDE_DIRS})
endif()

if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(v2x_gtsam_viewer PRIVATE nlohmann_json::nlohmann_json)
endif()
