# Build OBS-plugin
cmake_minimum_required(VERSION 3.16)

project (obs-backscrub)

find_package(LibObs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Define BACKSCRUB to build against an external copy..
if(DEFINED BACKSCRUB)
        set(CMAKE_MODULE_PATH "${BACKSCRUB}" CACHE PATH "Location of FindBackscrub.cmake")
        find_package(Backscrub REQUIRED)
        if(NOT BACKSCRUB_FOUND)
                message(FATAL_ERROR "Missing Backscrub")
        endif()
else()
        # assume backscrub is a sub-module in git
        find_package(Git REQUIRED)
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
        add_subdirectory(backscrub ${CMAKE_CURRENT_BINARY_DIR}/backscrub EXCLUDE_FROM_ALL)
        # TODO: export include paths from backscrub
        set(BACKSCRUB_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/backscrub)
        set(BACKSCRUB_LIBS backscrub)
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
