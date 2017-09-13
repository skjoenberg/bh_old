cmake_minimum_required(VERSION 2.8)

set(VE_FORTRAN true CACHE BOOL "VE-FORTRAN: Build the Openmpversial code generator.")
if(NOT VE_FORTRAN)
    return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

file(GLOB SRC *.cpp)

add_library(bh_ve_fortran SHARED ${SRC})

target_link_libraries(bh_ve_fortran bh)

install(TARGETS bh_ve_fortran DESTINATION ${LIBDIR} COMPONENT bohrium)


#
#   The rest of the this file is finding the compiler and flags to write in the config file
#

include(CheckCCompilerFlag)
include(FeatureSummary)

# Import and detect Fortran features
find_package(Fortran COMPONENTS bohrium)
set_package_properties(Fortran PROPERTIES TYPE RECOMMENDED PURPOSE "Multicore processing, essential for performance of the CPU VE.")
if(FORTRAN_FOUND)
    # Check for the SIMD flag
    set(Fortran_SIMD_C_FLAGS "${Fortran_C_FLAGS} ${Fortran_C_FLAGS}-simd")
    check_c_compiler_flag("${Fortran_SIMD_C_FLAGS}" FORTRAN_SIMD_FLAG_FOUND)
    # If the fortran-simd flag is supported, we should use it
    if(FORTRAN_SIMD_FLAG_FOUND)
        set(Fortran_C_FLAGS "${Fortran_SIMD_C_FLAGS}")
    endif()

    # Check for SIMD support
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_C_FLAGS} ${Fortran_C_FLAGS}")
    check_c_source_compiles("
    #include <omp.h>
    int main() {
      int i;
      int sum=42;
      #pragma omp parallel for simd reduction(+:sum)
      for(i=0; i<100; ++i)
        sum += i;
      return 0;
    }
    " FORTRAN_SIMD_FOUND)
    unset(CMAKE_REQUIRED_FLAGS)
endif()

# Check highly RECOMMENDED flags
check_c_compiler_flag(-O3 FLAG_03_FOUND)
check_c_compiler_flag(-march=native FLAG_MARCH_NATIVE_FOUND)
check_c_compiler_flag(-Werror FLAG_WERROR_FOUND)

#
# JIT-compiler capabilities: optimization, and parallelization
#
if(APPLE)
    #
    # Fallback on APPLE, this will run but without optimizations and parallelization.
    #
    set(VE_FORTRAN_COMPILER_INC "")
    set(VE_FORTRAN_COMPILER_FLG "-x c -dynamiclib -arch x86_64")
    set(VE_FORTRAN_COMPILER_LIB "-lm -L${VE_FORTRAN_COMPILER_INC}/${LIBDIR} -lbh")
    set(VE_FORTRAN_COMPILER_EXT "")
else()
    #
    # Mandatory flags
    set(VE_FORTRAN_COMPILER_FLG "-x c -fPIC -shared ${C99_FLAG}")
    #
    # Optimizations
    if (FLAG_03_FOUND)                                                      # Optimization Level
        set(VE_FORTRAN_COMPILER_FLG "${VE_FORTRAN_COMPILER_FLG} -O3")
    endif()
    if (FLAG_MARCH_NATIVE_FOUND)                                            # Machine specific code
        set(VE_FORTRAN_COMPILER_FLG "${VE_FORTRAN_COMPILER_FLG} -march=native")
    endif()
    if (FLAG_WERROR_FOUND)                                                  # Optimization Level
        set(VE_FORTRAN_COMPILER_FLG "${VE_FORTRAN_COMPILER_FLG} -Werror")
    endif()
    #
    # Parallelization
    #
    if(FORTRAN_FOUND)
        set(VE_FORTRAN_COMPILER_FLG "${VE_FORTRAN_COMPILER_FLG} ${Fortran_C_FLAGS}") # Fortran
    endif()
endif()

set(VE_FORTRAN_COMPILER_CMD         "${CMAKE_C_COMPILER}"                    CACHE STRING "VE_FORTRAN: JIT-Compiler")
set(VE_FORTRAN_COMPILER_INC         "-I${CMAKE_INSTALL_PREFIX}/share/bohrium/include")
set(VE_FORTRAN_COMPILER_INC         "${VE_FORTRAN_COMPILER_INC}"              CACHE STRING "VE_FORTRAN: JIT-Compiler includes")
set(VE_FORTRAN_COMPILER_LIB         "-lm -L${CMAKE_INSTALL_PREFIX}/${LIBDIR} -lbh" CACHE STRING "VE_FORTRAN: JIT-Compiler libraries")
set(VE_FORTRAN_COMPILER_FLG         "${VE_FORTRAN_COMPILER_FLG}"              CACHE STRING "VE_FORTRAN: JIT-Compiler flags")
set(VE_FORTRAN_COMPILER_FORTRAN      ${FORTRAN_FOUND}                          CACHE BOOL   "VE_FORTRAN: JIT-Compiler use Fortran")
set(VE_FORTRAN_COMPILER_FORTRAN_SIMD ${FORTRAN_SIMD_FOUND}                     CACHE BOOL   "VE_FORTRAN: JIT-Compiler use Fortran-SIMD")

# We need to cleanup the variables for the config file
if(VE_FORTRAN_COMPILER_FORTRAN)
    set(_VE_FORTRAN_COMPILER_FORTRAN "true" CACHE INTERNAL "config version")
else()
    set(_VE_FORTRAN_COMPILER_FORTRAN "false" CACHE INTERNAL "config version")
endif()
if(VE_FORTRAN_COMPILER_FORTRAN_SIMD)
    set(_VE_FORTRAN_COMPILER_FORTRAN_SIMD "true" CACHE INTERNAL "config version")
else()
    set(_VE_FORTRAN_COMPILER_FORTRAN_SIMD "false" CACHE INTERNAL "config version")
endif()
