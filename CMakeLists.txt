# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-backscrub)

find_package(LibObs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

if(NOT DEFINED BACKSCRUB)
        set(BACKSCRUB ../backscrub)
endif()
set(CMAKE_MODULE_PATH "${BACKSCRUB}" CACHE PATH "Location of FindBackscrub.cmake")
find_package(Backscrub REQUIRED)
if(NOT BACKSCRUB_FOUND)
        message(FATAL_ERROR "Missing Backscrub")
endif()

set(obs-backscrub_SOURCES
        obs-backscrub.cpp)

add_library(obs-backscrub MODULE
        ${obs-backscrub_SOURCES})
# Remove the 'lib' prefix for this target
set_target_properties(obs-backscrub PROPERTIES PREFIX "")

target_include_directories(obs-backscrub
        PUBLIC ${BACKSCRUB_INCLUDE})

target_link_libraries(obs-backscrub
        ${LIBOBS_LIBRARIES}
        ${BACKSCRUB_LIBS}
        ${OpenCV_LIBS})
