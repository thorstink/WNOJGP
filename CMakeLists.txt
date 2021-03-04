
cmake_minimum_required(VERSION 2.6)
enable_testing()
project(wnojgp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
find_package(gpmp2 REQUIRED) # Uses installed package

include_directories(${GTSAM_INCLUDE_DIR})
set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
# set(GTSAM_LIBRARIES gpmp2)   # TODO: automatic search libs

find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# for unittest scripts
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${GTSAM_DIR}/../GTSAMCMakeTools")

# Boost - same requirement as gtsam
find_package(Boost 1.50 REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(Boost COMPONENTS serialization REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

include_directories(include)

install(DIRECTORY include/wnoj_gp DESTINATION wnoj_gp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fsanitize=address -fno-omit-frame-pointer")
# add_subdirectory(test)
gtsamAddTestsGlob(gp "test/dotest.cc" "" gtsam)
add_executable(main test/main.cc)
target_link_libraries(main gtsam gpmp2)
# Install config and export files
# GtsamMakeConfigFile(gpmp2)
# export(TARGETS ${GPMP2_EXPORTED_TARGETS} FILE gpmp2-exports.cmake)