#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <minizip/unzip.h>
#include <minizip/zip.h>
#include <dirent.h>
#include <sys/stat.h>
//#include <libgen.h>
#include <functional>
#include <Utils.h>

using namespace std;

namespace ZipUtils
{
//-----------------------------------------------------------------------------
static bool zip_add_dir(zipFile zfile, std::string dirname)
{
    char*  temp;
    size_t len;
    int    ret;

    if (zfile == nullptr || dirname.empty())
        return false;

    len  = dirname.size();
    temp = new char[len + 2];
    memcpy(temp, dirname.c_str(), len);
    if (temp[len - 1] != '/')
    {
        temp[len]     = '/';
        temp[len + 1] = '\0';
    }
    else
    {
        temp[len] = '\0';
    }

    ret = zipOpenNewFileInZip64(zfile,
                                temp,
                                NULL,
                                NULL,
                                0,
                                NULL,
                                0,
                                NULL,
                                0,
                                0,
                                0);

    delete temp;
    if (ret != ZIP_OK)
        return false;

    zipCloseFileInZip(zfile);
    return true;
}
static bool zip_add_file(zipFile zfile, std::ifstream& fs, std::string filename, std::string zipPath = "")
{
    size_t size = (size_t)Utils::getFileSize(fs);

    zipPath = Utils::unifySlashes(zipPath);
    int ret = zipOpenNewFileInZip64(zfile, (zipPath + filename).c_str(), NULL, NULL, 0, NULL, 0, NULL,
            Z_DEFLATED, Z_DEFAULT_COMPRESSION, (size > 0xffffffff)?1:0);

    if (ret != ZIP_OK)
    {
        zipClose(zfile, nullptr);
        return false;
    }

    char   buf[8192];
    size_t n;
    while ((n = fs.readsome(buf, sizeof(buf))) > 0)
    {
        ret = zipWriteInFileInZip(zfile, buf, (unsigned int)n);
        if (ret != ZIP_OK)
        {
            zipCloseFileInZip(zfile);
            return false;
        }
    }
    zipCloseFileInZip(zfile);
    return true;
}
static bool zip_add_file(zipFile zfile, std::string filepath, std::string zipPath = "")
{
    std::ifstream fs(filepath, std::ios::binary);
    if (fs.fail())
    {
        return false;
    }

    return zip_add_file(zfile, fs, Utils::getFileName(filepath), zipPath);
}
//-----------------------------------------------------------------------------
bool unzip(std::string                                                 zipfile,
           std::function<bool(std::string path, std::string filename)> processFile,
           std::function<bool(const char* data, size_t len)>           writeChunk,
           std::function<bool(std::string path)>                       processDir)
{
    unzFile uzfile;
    bool    ret = true;
    size_t  n;
    char    name[256];

    uzfile = unzOpen64(zipfile.c_str());
    if (uzfile == NULL)
        return false;

    do
    {
        unz_file_info64 finfo;
        unsigned char   buf[8192];
        if (unzGetCurrentFileInfo64(uzfile,
                                    &finfo,
                                    name,
                                    sizeof(name),
                                    NULL,
                                    0,
                                    NULL,
                                    0) != UNZ_OK)
        {
            ret = false;
            break;
        }

        std::string dirname  = Utils::getDirName(Utils::trimRightString(name, "/"));
        std::string filename = Utils::getFileName(Utils::trimRightString(name, "/"));

        processDir(dirname);

        if (finfo.uncompressed_size == 0 && strlen(name) > 0 && name[strlen(name) - 1] == '/')
        {
            if (unzGoToNextFile(uzfile) != UNZ_OK)
                break;
            continue;
        }

        if (unzOpenCurrentFile(uzfile) != UNZ_OK)
        {
            ret = false;
            unzCloseCurrentFile(uzfile);
            break;
        }

        if (processFile(dirname, filename))
        {
            while ((n = unzReadCurrentFile(uzfile, buf, sizeof(buf))) > 0)
            {
                if (!writeChunk((const char*)buf, n))
                {
                    unzCloseCurrentFile(uzfile);
                    ret = false;
                    break;
                }
            }

            writeChunk(nullptr, 0);
            if (n < 0)
            {
                unzCloseCurrentFile(uzfile);
                ret = false;
                break;
            }
        }

        unzCloseCurrentFile(uzfile);
        if (unzGoToNextFile(uzfile) != UNZ_OK)
            break;

    } while (1);

    unzClose(uzfile);
    return ret;
}
//-----------------------------------------------------------------------------
bool zip(std::string path, std::string zipname)
{
    path = Utils::trimRightString(path, "/");

    if (zipname.empty())
        zipname = path + ".zip";

    zipFile zfile = zipOpen64(zipname.c_str(), 0);

    if (zfile == nullptr)
    {
        zipClose(zfile, nullptr);
        return false;
    }

    bool ret = true;
    string zipRootPath = Utils::getDirName(path);

    Utils::loopFileSystemRec(
      path,
      [zfile, &ret, zipRootPath](string path, string baseName, int depth) -> void {
          ret = ret && zip_add_file(zfile, path + baseName, path.erase(0, zipRootPath.size()));
      },
      [zfile, &ret, zipRootPath](string path, string baseName, int depth) -> void {
          ret = ret && zip_add_dir(zfile, path.erase(0, zipRootPath.size()) + baseName);
      },
      0);

    if (!ret)
    {
        free(zfile);
        Utils::removeFile(zipname);
        return false;
    }

    zipClose(zfile, NULL);
    return true;
}
//-----------------------------------------------------------------------------
bool unzip(std::string path, std::string dest, bool override)
{
    std::ofstream fs;

    dest = Utils::unifySlashes(dest);
    unzip(
      path,
      [&fs, &override, dest](std::string path, std::string filename) -> bool {
          if (override || !Utils::fileExists(dest + path + filename))
          {
              fs.open(dest + path + filename, std::ios::binary);
              return true;
          }
          return false;
      },
      [&fs](const char* data, size_t len) -> bool {
          if (data != nullptr)
          {
              try
              {
                  fs.write(data, len);
              }
              catch (std::exception& e)
              {
                  return false;
              }
          }
          else
              fs.close();
          return true;
      },
      [dest](std::string path) -> bool {
          if (!Utils::dirExists(dest + path))
              return Utils::makeDir(dest + path);
          return true;
      });
    return true;
}
//-----------------------------------------------------------------------------
}
