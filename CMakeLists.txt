# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-play)

find_package(LibObs REQUIRED)
find_package(OpenCV REQUIRED)

set(LIBDEEPSEG_LIBRARY ../backscrub/bin/libbackscrub.a)
set(LIBDEEPSEG_INCLUDE_DIR ../backscrub)

set(obs-backscrub_SOURCES
        obs-backscrub.cpp)

add_library(obs-backscrub MODULE
        ${obs-backscrub_SOURCES})
# Remove the 'lib' prefix for this target
set_target_properties(obs-backscrub PROPERTIES PREFIX "")

target_include_directories(obs-backscrub
        PUBLIC ${LIBDEEPSEG_INCLUDE_DIR})

get_filename_component(LIBDEEPSEG_ABS_LIB ${LIBDEEPSEG_LIBRARY} ABSOLUTE)
target_link_libraries(obs-backscrub
        ${LIBOBS_LIBRARIES}
        ${LIBDEEPSEG_ABS_LIB}
        ${OpenCV_LIBS})
