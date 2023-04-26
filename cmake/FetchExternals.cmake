#
# Fetch external libraries from remote repositories
#

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.3
)

FetchContent_MakeAvailable(eigen)