cmake_minimum_required(VERSION 3.20)
project(creature_runtime CXX)
# Creature runtime: C++ core + C API + console tests + optional Godot GDExtension (class DogBrain).
# No absolute paths here. Dependencies come from -D options or environment variables (see README.md).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  add_compile_options(/utf-8 /EHsc /W3 /permissive-)
  add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

option(CREATURE_BUILD_TESTS "Build the console tests" ON)
option(CREATURE_BUILD_GODOT "Build the Godot GDExtension (needs GODOT_CPP_DIR)" OFF)
set(CREATURE_ADDON_BIN "${CMAKE_CURRENT_SOURCE_DIR}/../sandboxes/godot_creature/addons/dogbrain/bin" CACHE PATH
    "Folder that receives the GDExtension library and the ONNX Runtime library after the Godot build")
set(CREATURE_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models/spider_v3" CACHE PATH "Model folder used by ctest")

# Build outputs stay inside the build folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---------------------------------------------------------------- ONNX Runtime
# ONNXRUNTIME_DIR: the unpacked official release (include/, lib/) or the NuGet layout (build/native/include, runtimes/<rid>/native).
if(NOT ONNXRUNTIME_DIR AND DEFINED ENV{ONNXRUNTIME_DIR})
  set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}")
endif()
if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_DIR (-DONNXRUNTIME_DIR=<folder> or the environment variable). tools/fetch_deps.py downloads it.")
endif()
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_DIR}/include" "${ONNXRUNTIME_DIR}/build/native/include" NO_DEFAULT_PATH)
find_library(ORT_LIBRARY NAMES onnxruntime
  HINTS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/runtimes/win-x64/native" "${ONNXRUNTIME_DIR}/runtimes/linux-x64/native" NO_DEFAULT_PATH)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_DIR} (need onnxruntime_cxx_api.h and the onnxruntime library)")
endif()
set(ORT_RUNTIME_FILES "")
if(WIN32)
  find_file(ORT_DLL onnxruntime.dll
    HINTS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/runtimes/win-x64/native" NO_DEFAULT_PATH)
  if(NOT ORT_DLL)
    message(FATAL_ERROR "onnxruntime.dll not found in ${ONNXRUNTIME_DIR}")
  endif()
  set(ORT_RUNTIME_FILES "${ORT_DLL}")
else()
  get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
  file(GLOB ORT_RUNTIME_FILES "${ORT_LIB_DIR}/libonnxruntime.so*")
endif()
add_library(ort::ort UNKNOWN IMPORTED)
set_target_properties(ort::ort PROPERTIES IMPORTED_LOCATION "${ORT_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

# ---------------------------------------------------------------- nlohmann json
# NLOHMANN_JSON_DIR: an include folder that holds nlohmann/json.hpp. Without it, FetchContent downloads release 3.12.0.
if(NOT NLOHMANN_JSON_DIR AND DEFINED ENV{NLOHMANN_JSON_DIR})
  set(NLOHMANN_JSON_DIR "$ENV{NLOHMANN_JSON_DIR}")
endif()
if(NLOHMANN_JSON_DIR)
  if(NOT EXISTS "${NLOHMANN_JSON_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "NLOHMANN_JSON_DIR must contain nlohmann/json.hpp: ${NLOHMANN_JSON_DIR}")
  endif()
  add_library(nlohmann_json_local INTERFACE)
  target_include_directories(nlohmann_json_local INTERFACE "${NLOHMANN_JSON_DIR}")
  add_library(creature::json ALIAS nlohmann_json_local)
else()
  include(FetchContent)
  FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
  add_library(creature::json ALIAS nlohmann_json)
endif()

# ---------------------------------------------------------------- core + C API
add_library(creature_core STATIC src/dog_brain_core.cpp)
target_include_directories(creature_core PUBLIC src)
target_link_libraries(creature_core PUBLIC ort::ort creature::json)

add_library(creature_runtime SHARED src/creature_runtime.cpp)
target_compile_definitions(creature_runtime PRIVATE CREATURE_RUNTIME_BUILD PUBLIC CREATURE_RUNTIME_SHARED)
target_include_directories(creature_runtime PUBLIC src)
target_link_libraries(creature_runtime PRIVATE creature_core)
if(NOT WIN32)
  set_target_properties(creature_runtime PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON BUILD_RPATH "$ORIGIN")
endif()

# Put the ONNX Runtime library next to a target (Windows needs it beside the exe / dll).
function(creature_copy_ort target)
  if(ORT_RUNTIME_FILES)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_RUNTIME_FILES} $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()
creature_copy_ort(creature_runtime)

if(CREATURE_BUILD_TESTS)
  enable_testing()
  add_executable(dog_brain_test test/dog_brain_test.cpp)
  target_link_libraries(dog_brain_test PRIVATE creature_core)
  creature_copy_ort(dog_brain_test)
  if(NOT WIN32)
    set_target_properties(dog_brain_test PROPERTIES BUILD_RPATH "$ORIGIN")
  endif()
  add_executable(creature_c_api_test test/creature_c_api_test.cpp)
  target_link_libraries(creature_c_api_test PRIVATE creature_runtime)
  creature_copy_ort(creature_c_api_test)
  if(NOT WIN32)
    set_target_properties(creature_c_api_test PROPERTIES BUILD_RPATH "$ORIGIN")
  endif()
  add_test(NAME dog_brain_test COMMAND dog_brain_test "${CREATURE_TEST_MODEL_DIR}" 2 walk 1.5 fast 3.5)
  add_test(NAME creature_c_api_test COMMAND creature_c_api_test "${CREATURE_TEST_MODEL_DIR}" 2 1.5)
endif()

# ---------------------------------------------------------------- Godot GDExtension (optional)
# GODOT_CPP_DIR: a godot-cpp checkout (-D or environment variable).
#   Source mode (default): add_subdirectory(GODOT_CPP_DIR), godot-cpp builds with this project.
#   Prebuilt mode: GODOT_CPP_LIB = a godot-cpp static library built before (same MSVC runtime: /MD).
#     Headers: GODOT_CPP_DIR/include + GODOT_CPP_GEN_DIR/include (generated bindings; default GODOT_CPP_DIR/gen).
if(CREATURE_BUILD_GODOT)
  if(NOT GODOT_CPP_DIR AND DEFINED ENV{GODOT_CPP_DIR})
    set(GODOT_CPP_DIR "$ENV{GODOT_CPP_DIR}")
  endif()
  if(NOT GODOT_CPP_DIR)
    message(FATAL_ERROR "CREATURE_BUILD_GODOT=ON needs GODOT_CPP_DIR (-D or environment variable)")
  endif()
  if(NOT GODOT_CPP_LIB AND DEFINED ENV{GODOT_CPP_LIB})
    set(GODOT_CPP_LIB "$ENV{GODOT_CPP_LIB}")
  endif()
  if(GODOT_CPP_LIB)
    if(NOT GODOT_CPP_GEN_DIR)
      set(GODOT_CPP_GEN_DIR "${GODOT_CPP_DIR}/gen")
    endif()
    add_library(godot-cpp STATIC IMPORTED)
    set(_gd_defs GDEXTENSION THREADS_ENABLED TYPED_METHOD_BIND NOMINMAX)
    if(WIN32)
      list(APPEND _gd_defs WINDOWS_ENABLED)
    else()
      list(APPEND _gd_defs LINUX_ENABLED UNIX_ENABLED)
    endif()
    set_target_properties(godot-cpp PROPERTIES IMPORTED_LOCATION "${GODOT_CPP_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${GODOT_CPP_GEN_DIR}/include;${GODOT_CPP_DIR}/include"
      INTERFACE_COMPILE_DEFINITIONS "${_gd_defs}")
    if(MSVC)
      set_property(TARGET godot-cpp APPEND PROPERTY INTERFACE_COMPILE_OPTIONS /Zc:__cplusplus)
      set_property(TARGET godot-cpp APPEND PROPERTY INTERFACE_LINK_OPTIONS /MANIFEST:NO)
    endif()
    set(GODOT_CPP_TARGET godot-cpp)
  else()
    add_subdirectory("${GODOT_CPP_DIR}" godot-cpp-build)
    if(TARGET godot::cpp)
      set(GODOT_CPP_TARGET godot::cpp)
    else()
      set(GODOT_CPP_TARGET godot-cpp)
    endif()
  endif()

  add_library(dogbrain_gdext SHARED godot/dog_brain_gd.cpp)
  target_link_libraries(dogbrain_gdext PRIVATE creature_core ${GODOT_CPP_TARGET})
  # Library name matches the .gdextension file: dogbrain.dll on Windows, libdogbrain.so on Linux.
  set_target_properties(dogbrain_gdext PROPERTIES OUTPUT_NAME "dogbrain")
  if(WIN32)
    set_target_properties(dogbrain_gdext PROPERTIES PREFIX "")
  else()
    set_target_properties(dogbrain_gdext PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
  endif()
  # Copy step: the extension library + the ONNX Runtime library go to CREATURE_ADDON_BIN.
  add_custom_command(TARGET dogbrain_gdext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CREATURE_ADDON_BIN}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:dogbrain_gdext> ${ORT_RUNTIME_FILES} "${CREATURE_ADDON_BIN}")
endif()
