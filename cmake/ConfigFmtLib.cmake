# Adapted from https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
#
# CAVEAT: use DownloadProject.cmake
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(DownloadProject)
download_project(PROJ                fmt
                 GIT_REPOSITORY      https://github.com/fmtlib/fmt.git
                 GIT_TAG             master
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR})

include_directories("${fmt_SOURCE_DIR}/include")
