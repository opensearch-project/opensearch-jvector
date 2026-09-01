#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.24)
project(faiss_c_native)

include(FetchContent)

set(FAISS_ENABLE_C_API ON CACHE BOOL "Enable C API" FORCE)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)

FetchContent_Declare(
    faiss
    GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
    GIT_TAG        v1.11.0
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(faiss)
