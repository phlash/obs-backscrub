# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-play)

find_package(LibObs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Pull in backscrub and dependencies
include(../backscrub/build/BackscrubTargets.cmake)
set(LIBBACKSCRUB_INCLUDE_DIR ../backscrub)

set(obs-backscrub_SOURCES
        obs-backscrub.cpp)

add_library(obs-backscrub MODULE
        ${obs-backscrub_SOURCES})
# Remove the 'lib' prefix for this target
set_target_properties(obs-backscrub PROPERTIES PREFIX "")

target_include_directories(obs-backscrub
		PUBLIC ${LIBBACKSCRUB_INCLUDE_DIR})

target_link_libraries(obs-backscrub
        ${LIBOBS_LIBRARIES}
		backscrub
        ${OpenCV_LIBS})
