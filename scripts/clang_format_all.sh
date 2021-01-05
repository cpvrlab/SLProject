# runs clang-format recursively on defined directories

find ../modules/test0 ../modules/testX \
	-type f \( -iname \*.cpp -o -iname \*.h -o -iname \*.hpp -o -iname \*.mm -o -iname \*.m \) \
	-not -path '*externals*' \
	-exec clang-format -i {} \;