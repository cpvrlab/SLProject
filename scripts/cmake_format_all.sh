# runs cmake-format recursively on defined directories
# (you have to install cmake-format from https://github.com/cheshirekow/cmake_format)

SL_ROOT="$(dirname -- "$0")/.."
SEARCH_PATHS="$SL_ROOT"

find $SEARCH_PATHS \
	-type f \( -iname CMakeLists.txt -o -iname \*.cmake \) \
	-not -path '*externals*' \
	-not -path '*experimental*' \
	-exec cmake-format -i {} \;