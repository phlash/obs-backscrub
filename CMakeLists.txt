# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-play)

find_package(LibObs REQUIRED)
message("LIBOBJS_INCLUDE_DIRS:" ${LIBOBS_INCLUDE_DIRS})
message("LIBOBJS_LIBRARIES:" ${LIBOBS_LIBRARIES})

set(obs-play_SOURCES
        obs-play.cpp)

add_library(obs-play MODULE
        ${obs-play_SOURCES})

target_link_libraries(obs-play
        ${LIBOBS_LIBRARIES})
