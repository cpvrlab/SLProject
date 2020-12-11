#ifndef CPLVRLAB_ZIP_UTILS_H
#define CPLVRLAB_ZIP_UTILS_H

#include <string>
#include <functional>

namespace ZipUtils
{
bool zip(std::string path, std::string zipname = "");

bool unzip(std::string                                                 zipfile,
           std::function<bool(std::string path, std::string filename)> processFile,
           std::function<bool(const char* data, size_t len)>           writeChunk,
           std::function<bool(std::string path)>                       processDir);

bool unzip(std::string path, std::string dest = "", bool override = true);
}
#endif
