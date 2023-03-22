//#############################################################################
//  File:      ZipUtils.cpp
//  Authors:   Luc Girod
//  Date:      2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <cstring>
#include <string>
#include <iostream>
#include <functional>
#include <Utils.h>
#include <minizip/unzip.h>
#include <minizip/zip.h>

namespace ZipUtils
{
//-----------------------------------------------------------------------------
/*!
 *
 * @param zfile ???
 * @param dirname ???
 * @return ???
 */
static bool zip_add_dir(zipFile zfile, string dirname)
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

    delete[] temp;
    if (ret != ZIP_OK)
        return false;

    zipCloseFileInZip(zfile);
    return true;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param zfile ???
 * @param fs ???
 * @param filename ???
 * @param zipPath ???
 * @return ???
 */
static bool zip_add_file(zipFile        zfile,
                         std::ifstream& fs,
                         string         filename,
                         string         zipPath = "")
{
    size_t size = (size_t)Utils::getFileSize(fs);

    zipPath = Utils::unifySlashes(zipPath);
    int ret = zipOpenNewFileInZip64(zfile,
                                    (zipPath + filename).c_str(),
                                    NULL,
                                    NULL,
                                    0,
                                    NULL,
                                    0,
                                    NULL,
                                    Z_DEFLATED,
                                    Z_DEFAULT_COMPRESSION,
                                    (size > 0xffffffff) ? 1 : 0);

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
//-----------------------------------------------------------------------------
/*!
 *
 * @param zfile ???
 * @param filepath ???
 * @param zipPath ???
 * @return ???
 */
static bool zip_add_file(zipFile zfile,
                         string  filepath,
                         string  zipPath = "")
{
    std::ifstream fs(filepath, std::ios::binary);
    if (fs.fail())
    {
        return false;
    }

    return zip_add_file(zfile,
                        fs,
                        Utils::getFileName(filepath),
                        zipPath);
}
//-----------------------------------------------------------------------------
/*!
 *
 @param zipfile ???
 @param processFile ???
 @param writeChunk ???
 @param processDir ???
 @param progress Progress function to call for progress visualization
 @return ???
 */
bool unzip(string                                         zipfile,
           function<bool(string path, string filename)>   processFile,
           function<bool(const char* data, size_t len)>   writeChunk,
           function<bool(string path)>                    processDir,
           function<int(int currentFile, int totalFiles)> progress = nullptr)
{
    unzFile uzfile;
    bool    ret = true;
    size_t  n;
    char    name[256];
    int     nbProcessedFile = 0;

    uzfile = unzOpen64(zipfile.c_str());
    if (uzfile == NULL)
        return false;

    // Get info about the zip file
    unz_global_info global_info;
    if (unzGetGlobalInfo(uzfile, &global_info) != UNZ_OK)
    {
        unzClose(uzfile);
        return (bool)-1;
    }

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

        string dirname  = Utils::getDirName(Utils::trimRightString(name, "/"));
        string filename = Utils::getFileName(Utils::trimRightString(name, "/"));

        processDir(dirname);
        if (progress != nullptr && progress(nbProcessedFile++, (int)global_info.number_entry))
        {
            unzClose(uzfile);
            return false;
        }

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

    if (progress != nullptr)
        progress((int)global_info.number_entry, (int)global_info.number_entry);
    return ret;
}
//-----------------------------------------------------------------------------
/*!
 *
 * @param path ???
 * @param zipname ???
 * @return ???
 */
bool zip(string path, string zipname)
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

    bool   ret         = true;
    string zipRootPath = Utils::getDirName(path);

    Utils::loopFileSystemRec(
      path,
      [zfile,
       &ret,
       zipRootPath](string path,
                    string baseName,
                    int    depth) -> void
      {
          ret = ret && zip_add_file(zfile,
                                    path + baseName,
                                    path.erase(0,
                                               zipRootPath.size()));
      },
      [zfile, &ret, zipRootPath](string path,
                                 string baseName,
                                 int    depth) -> void
      {
          ret = ret && zip_add_dir(zfile,
                                   path.erase(0, zipRootPath.size()) + baseName);
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
/*!
 Unzips a zip file
 @param path ???
 @param dest ???
 @param override Overrides existing files on destination
 @param progress Progress function to call for progress visualization
 @return Returns true on success
 */
bool unzip(string                                         path,
           string                                         dest,
           bool                                           override,
           function<int(int currentFile, int totalFiles)> progress)
{
    std::ofstream fs;

    dest = Utils::unifySlashes(dest);
    unzip(
      path,
      [&fs, &override, dest](string path, string filename) -> bool
      {
          if (override || !Utils::fileExists(dest + path + filename))
          {
              fs.open(dest + path + filename, std::ios::binary);
              return true;
          }
          return false;
      },
      [&fs](const char* data, size_t len) -> bool
      {
          if (data != nullptr)
          {
              try
              {
                  fs.write(data, len);
              }
              catch (std::exception& e)
              {
                  std::cout << e.what() << std::endl;
                  return false;
              }
          }
          else
              fs.close();
          return true;
      },
      [dest](string path) -> bool
      {
          if (!Utils::dirExists(dest + path))
              return Utils::makeDir(dest + path);
          return true;
      },
      progress);
    return true;
}
//-----------------------------------------------------------------------------
}
