#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <minizip/unzip.h>
#include <minizip/zip.h>
#include <dirent.h>
#include <sys/stat.h>
#include <libgen.h>
#include <functional>
#include <Utils.h>

using namespace std;

namespace ZipUtils
{

static bool zip_add_dir(zipFile zfile, std::string dirname)
{
    char   *temp;
    size_t  len;
    int     ret;

    if (zfile == nullptr || dirname.empty())
        return false; 

    len = dirname.size();
    temp = new char[len+2];
    memcpy(temp, dirname.c_str(), len);
    if (temp[len-1] != '/')
    {
        temp[len] = '/';
        temp[len+1] = '\0';
    }
    else 
    {
        temp[len] = '\0';
    }

    ret = zipOpenNewFileInZip64(zfile, temp, NULL, NULL, 0, NULL, 0, NULL, 0, 0, 0);

    delete temp;
    if (ret != ZIP_OK)
        return false;

    zipCloseFileInZip(zfile);
    return true;
}

static bool zip_add_file(zipFile zfile, std::ifstream &fs, std::string filename)
{
    size_t size = (size_t)Utils::getFileSize(fs);

    int ret = zipOpenNewFileInZip64(zfile, filename.c_str(), NULL, NULL, 0, NULL, 0, NULL,
            Z_DEFLATED, Z_DEFAULT_COMPRESSION, (size > 0xffffffff)?1:0);

    if (ret != ZIP_OK)
    {
        zipClose(zfile, nullptr);
        return false;
    }

    char buf[8192];
    size_t n;
    while ((n = fs.readsome(buf, sizeof(buf))) > 0) 
    {
        ret = zipWriteInFileInZip(zfile, buf, n);
        if (ret != ZIP_OK) 
        {
            zipCloseFileInZip(zfile);
            return false;
        }
    }
    zipCloseFileInZip(zfile);
    return true;
}

static bool zip_add_file(zipFile zfile, std::string filename)
{
    std::ifstream fs(filename, std::ios::binary);
    if (fs.fail())
    {
        return false;
    }

    return zip_add_file(zfile, fs, filename);
}

bool unzip(std::string zipfile,
           std::function<bool(std::string path, std::string filename)> processFile,
           std::function<void(std::string path, std::string filename, const char* data, size_t len)> writeChunk,
           std::function<void(std::string path, std::string filename)> processDir)
{
    unzFile          uzfile;
    bool             ret = true;
    size_t           n;
    char             name[256];

    uzfile = unzOpen64(zipfile.c_str());
    if (uzfile == NULL)
        return false;

    do {
        unsigned char   buf[8192];
        unz_file_info64 finfo;
        if (unzGetCurrentFileInfo64(uzfile, &finfo, name, sizeof(name), NULL, 0, NULL, 0) != UNZ_OK)
        {
            ret = false;
            continue;
        }

        std::string dirname = Utils::getDirName(Utils::trimString(name, "/"));
        std::string filename = Utils::getFileName(Utils::trimString(name, "/"));

        if (finfo.uncompressed_size == 0 && strlen(name) > 0 && name[strlen(name)-1] == '/')
        {
            processDir(dirname, filename);
            unzGoToNextFile(uzfile);
            continue;
        }

        if (unzOpenCurrentFile(uzfile) != UNZ_OK)
        {
            ret = false;
            continue;
        }
        
        if (processFile(dirname, filename))
        {
            while ((n = unzReadCurrentFile(uzfile, buf, sizeof(buf))) > 0) 
            {
                writeChunk(dirname, filename, (const char*)buf, n);
            }

            writeChunk(dirname, filename, nullptr, 0);
            if (n < 0) {
                unzCloseCurrentFile(uzfile);
                ret = false;
                continue;
            }
        }

        unzCloseCurrentFile(uzfile);
        if (unzGoToNextFile(uzfile) != UNZ_OK)
            ret = false;

    } while (ret);

    unzClose(uzfile);
    return ret;
}

bool zip(std::string path, std::string zipname)
{
    if (zipname.empty())
        zipname = Utils::trimString(path, "/") + ".zip";

    zipFile zfile = zipOpen64(zipname.c_str(), 0);

    if (zfile == nullptr)
    {
        zipClose(zfile, nullptr);
        return false;
    }

    bool ret = true;

    Utils::loopFileSystemRec(path,
                      [zfile, &ret](string path, string baseName, int depth) -> void {
                           ret = ret && zip_add_file(zfile, path+baseName);
                      },
                      [zfile, &ret](string path, string baseName, int depth) -> void {
                           ret = ret && zip_add_dir(zfile, path+baseName);
                      }, 0);

    if (ret == false)
    {
        free(zfile);
        Utils::removeFile(zipname);
        return false;
    }
        
    zipClose(zfile, NULL);
    return true;
}

bool unzip(std::string path, std::string dest)
{
    std::ofstream fs;

    dest = Utils::unifySlashes(dest);

    unzip(
      path,
      [&fs, dest](std::string path, std::string filename) -> bool {
          if (!Utils::fileExists(dest + path + filename))
          {
              fs.open(dest + path + filename, std::ios::binary);
              return true;
          }
          return false;
      },
      [&fs](std::string path, std::string filename, const char* data, size_t len) -> void {
          if (data != nullptr)
              fs.write(data, len);
          else
              fs.close();
      },
      [dest](std::string path, std::string filename) -> void {
          Utils::makeDir(dest + path + filename);
      });
    return true;
}
}
