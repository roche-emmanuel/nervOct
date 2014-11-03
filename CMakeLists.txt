CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# The name of this project:
PROJECT(MXSight)
SET(PROJECT_VERSION 1.0)

# Include the config:
INCLUDE(cmake/Config.cmake)

SET(CMAKE_MODULE_PATH "cmake ${CMAKE_MODULE_PATH}")

ADD_SUBDIRECTORY(sources)

