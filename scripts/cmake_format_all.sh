# runs cmake-format recursively on defined directories
# (you have to install cmake-format from https://github.com/cheshirekow/cmake_format)

find ../modules/test0 ../modules/testX \
	-type f \( -iname CMakeLists.txt -o -iname \*.cmake \) \
	-not -path '*externals*' \
	-exec cmake-format -i {} \;