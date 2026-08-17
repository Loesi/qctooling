cmake_minimum_required(VERSION 3.15)
project(qctooling NONE)

include(FetchContent)

# -------------------------------------------------------------------------
# Pull and compile libcint while this package is being build/installed.
# Pinned to the v6.1.3 release tarball of htpps://github.com/sunqm/libcint.
# -------------------------------------------------------------------------
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(cint
    URL https://github.com/sunqm/libcint/archive/refs/tags/v6.1.3.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

# libcint builds as target `cint`. Disable its optional Fortran interface so
# no Fortran compiler is needed on build machine.
set(WITH_FORTRAN OFF CACHE BOOL "libcint Fortran interface" FORCE)

FetchContent_MakeAvailable(cint)

# -------------------------------------------------------------------------
# Ship the built shared library inside the package so it can be loaded at
# runtime with ctypes. On Linux/macOS this installs the versioned libcint
# library together with its SONAME namelink, on Windows the DLL (RUNTIME).
# -------------------------------------------------------------------------
install(TARGETS cint
    LIBRARY DESTINATION qctooling/_libcint
    RUNTIME DESTINATION qctooling/_libcint)
