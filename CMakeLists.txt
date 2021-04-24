# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-play)

find_package(LibObs REQUIRED)
find_package(OpenCV REQUIRED)

set(LIBDEEPSEG_LIBRARY ../deepbacksub/bin/libdeepseg.a)
set(LIBDEEPSEG_INCLUDE_DIR ../deepbacksub)

set(obs-play_SOURCES
        obs-play.cpp)

add_library(obs-play MODULE
        ${obs-play_SOURCES})

target_include_directories(obs-play
        PUBLIC ${LIBDEEPSEG_INCLUDE_DIR})

get_filename_component(LIBDEEPSEG_ABS_LIB ${LIBDEEPSEG_LIBRARY} ABSOLUTE)
target_link_libraries(obs-play
        ${LIBOBS_LIBRARIES}
        ${LIBDEEPSEG_ABS_LIB}
        ${OpenCV_LIBS})
