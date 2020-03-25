#ifndef APP_DIRECTORIES
#define APP_DIRECTORIES
#include <string>

typedef struct AppDirectories
{
    std::string writableDir;
    std::string waiDataRoot;
    std::string slDataRoot;
    std::string vocabularyDir;
    std::string logFileDir;

} AppDirectories;

#endif
