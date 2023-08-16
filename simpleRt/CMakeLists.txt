cmake_minimum_required ( VERSION 3.11 )

project ( RTWeekendCuda
  VERSION 2.95.0
  )

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # for multi-target pre-compilations
else()
  message(STATUS "No CUDA support")
endif()



# set ( HEADER_RAPIDJSON
#   Raytracing_CUDA/rapidjson/*.h
#   Raytracing_CUDA/rapidjson/error/*.h
#   Raytracing_CUDA/rapidjson/internal/*.h
#   Raytracing_CUDA/rapidjson/misinttypes/*.h
# )

# set ( SOURCE_ONE_WEEKEND_CUDA
#   ${HEADER_RAPIDJSON}
#   Raytracing_CUDA/*.cuh
#   Raytracing_CUDA/*.cu
# )

file(GLOB SOURCE_ONE_WEEKEND_CUDA "src/*.cuh" "src/*.cu" )


add_executable(exe_ray_cuda ${SOURCE_ONE_WEEKEND_CUDA})

target_include_directories(exe_ray_cuda  PRIVATE 
  src/rapidjson 
  src/rapidjson/error 
  src/rapidjson/internal
  src/rapidjson/misinttypes
  ../dependencies
  ../include
  ../dependencies/eigen
  ../dependencies/filesystem
  ../dependencies/nanovdb
  ../dependencies/tinylogger
  )
# target_include_directories(exe_simple_rt PUBLIC src)
