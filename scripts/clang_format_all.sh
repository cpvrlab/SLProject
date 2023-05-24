# runs clang-format recursively on defined directories

SL_ROOT="$(dirname -- "$0")/.."
SEARCH_PATHS="$SL_ROOT/modules/*/source $SL_ROOT/apps"

find $SEARCH_PATHS \
	-type f \( -iname \*.cpp -o -iname \*.h -o -iname \*.hpp -o -iname \*.mm -o -iname \*.m \) \
	-not -path '*externals*' \
	-not -path '*orb_slam*' \
	-exec clang-format -i {} \;