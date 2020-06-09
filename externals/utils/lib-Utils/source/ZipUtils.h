#ifndef CPLVRLAB_ZIP_UTILS_H
#define CPLVRLAB_ZIP_UTILS_H

#include <string>

namespace ZipUtils
{
bool zip(std::string path, std::string zipname = "");

bool unzip(std::string zipfile,
           std::function<bool(std::string path, std::string filename)> processFile,
           std::function<void(std::string path, std::string filename, const char* data, size_t len)> writeChunk,
           std::function<void(std::string path, std::string filename)> processDir);

bool unzip(std::string path, std::string dest = "");
}
#endif
