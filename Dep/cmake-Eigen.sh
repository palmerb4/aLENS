#!/bin/bash

SOURCE_PATH=../eigen-3.4.0

EXTRA_ARGS=$@

rm -f CMakeCache.txt

cmake \
    -D CMAKE_INSTALL_PREFIX:FILEPATH="$SFTPATH" \
    -D CMAKE_BUILD_TYPE:STRING="Release" \
    -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
    -D CMAKE_C_COMPILER:STRING="mpicc" \
    -D CMAKE_CXX_FLAGS:STRING="$CXXFLAGS" \
    -D CMAKE_C_FLAGS:STRING="$CFLAGS" \
    $EXTRA_ARGS \
    $SOURCE_PATH