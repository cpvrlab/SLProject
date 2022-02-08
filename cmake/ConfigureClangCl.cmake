#
# Add some extra compile options that are required when compiling using clang-cl on Windows
# clang-cl is a drop-in replacement for cl (the MSVC compiler)
#

# Check if the compiler is Clang and if it simulates MSVC
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND "${CMAKE_CXX_SIMULATE_ID}" MATCHES "MSVC")
    # clang-cl doesn't define the _WINDOWS macro
    add_compile_definitions(_WINDOWS)

    # NOMINMAX disables the macros "min" and "max" from a Windows header that clang-cl for some reason includes
    # These macros override std::min and std::max from the C++ standard library, which blows everything up
    add_compile_definitions(NOMINMAX)

    # Set additional flags
    # /EHs              Enable Exception handling (https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170)
    # -march=native     Enable AVX intrinsics for fbow (won't compile on machines without AVX support)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc -march=native")
endif ()