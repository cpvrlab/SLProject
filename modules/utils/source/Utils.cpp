// #############################################################################
//   File:      Utils.cpp
//   Authors:   Marcus Hudritsch
//   Date:      May 2019
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#include <Utils.h>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstdarg>
#include <cstring>
#include <utility>
#include <vector>
#include <algorithm>
#include <thread>

#ifndef __EMSCRIPTEN__
#    include <asio.hpp>
#    include <asio/ip/tcp.hpp>
#endif

#if defined(_WIN32)
#    if _MSC_VER >= 1912
#        define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#        include <experimental/filesystem>
#        define USE_STD_FILESYSTEM
namespace fs = std::experimental::filesystem;
#    else
#        include <direct.h> //_getcwd
#    endif
#elif defined(__APPLE__)
#    if defined(TARGET_OS_IOS) && (TARGET_OS_IOS == 1)
#        include "Utils_iOS.h"
#    endif
#    include <dirent.h>
#    include <sys/stat.h> //dirent
#    include <unistd.h>   //getcwd
#elif defined(ANDROID) || defined(ANDROID_NDK)
#    include <android/log.h>
#    include <dirent.h>
#    include <unistd.h> //getcwd
#    include <sys/stat.h>
#    include <sys/time.h>
#    include <sys/system_properties.h>
#elif defined(linux) || defined(__linux) || defined(__linux__)
#    include <dirent.h>
#    include <unistd.h> //getcwd
#    include <sys/types.h>
#    include <sys/stat.h>
#elif defined(__EMSCRIPTEN__)
#    include <emscripten.h>
#    include <dirent.h>
#    include <unistd.h>
#    include <sys/types.h>
#    include <sys/stat.h>
#endif

#ifndef __EMSCRIPTEN__
using asio::ip::tcp;
#endif

#ifdef __EMSCRIPTEN__
// clang-format off
EM_JS(void, alert, (const char* string), {
    alert(UTF8ToString(string));
})
// clang-format on
#endif

using std::fstream;

namespace Utils
{
///////////////////////////////
// Global variables          //
///////////////////////////////
std::unique_ptr<CustomLog> customLog;

///////////////////////////////
// String Handling Functions //
///////////////////////////////

//-----------------------------------------------------------------------------
// Returns a string from a float with max. one trailing zero
string toString(float f, int roundedDecimals)
{
    stringstream ss;
    ss << std::fixed << std::setprecision(roundedDecimals) << f;
    string num = ss.str();
    if (num == "-0.0") num = "0.0";
    return num;
}
//-----------------------------------------------------------------------------
// Returns a string from a double with max. one trailing zero
string toString(double d, int roundedDecimals)
{
    stringstream ss;
    ss << std::fixed << std::setprecision(roundedDecimals) << d;
    string num = ss.str();
    if (num == "-0.0") num = "0.0";
    return num;
}
//-----------------------------------------------------------------------------
// Returns a string in lower case
string toLowerString(string s)
{
    string cpy(std::move(s));
    transform(cpy.begin(), cpy.end(), cpy.begin(), ::tolower);
    return cpy;
}
//-----------------------------------------------------------------------------
// Returns a string in upper case
string toUpperString(string s)
{
    string cpy(std::move(s));
    transform(cpy.begin(), cpy.end(), cpy.begin(), ::toupper);
    return cpy;
}
//-----------------------------------------------------------------------------
// trims a string at both ends
string trimString(const string& s, const string& drop)
{
    string r = s;
    r        = r.erase(r.find_last_not_of(drop) + 1);
    return r.erase(0, r.find_first_not_of(drop));
}
//-----------------------------------------------------------------------------
// trims a string at the right end
string trimRightString(const string& s, const string& drop)
{
    string r = s;
    r        = r.erase(r.find_last_not_of(drop) + 1);
    return r;
}
//-----------------------------------------------------------------------------
// trims a string at the left end
string trimLeftString(const string& s, const string& drop)
{
    string r = s;
    r        = r.erase(r.find_first_not_of(drop) + 1);
    return r;
}
//-----------------------------------------------------------------------------
// Splits an input string at a delimeter character into a string vector
void splitString(const string&   s,
                 char            delimiter,
                 vector<string>& splits)
{
    string::size_type i = 0;
    string::size_type j = s.find(delimiter);

    while (j != string::npos)
    {
        splits.push_back(s.substr(i, j - i));
        i = ++j;
        j = s.find(delimiter, j);
        if (j == string::npos)
            splits.push_back(s.substr(i, s.length()));
    }
}
//-----------------------------------------------------------------------------
// Replaces in the source string the from string by the to string
void replaceString(string&       source,
                   const string& from,
                   const string& to)
{
    // Code from: http://stackoverflow.com/questions/2896600/
    // how-to-replace-all-occurrences-of-a-character-in-string
    string newString;
    newString.reserve(source.length()); // avoids a few memory allocations

    string::size_type lastPos = 0;
    string::size_type findPos = 0;

    while (string::npos != (findPos = source.find(from, lastPos)))
    {
        newString.append(source, lastPos, findPos - lastPos);
        newString += to;
        lastPos = findPos + from.length();
    }

    // Care for the rest after last occurrence
    newString += source.substr(lastPos);
    source.swap(newString);
}
//-----------------------------------------------------------------------------
// Returns a vector of string one per line of a multiline string
vector<string> getStringLines(const string& multiLineString)
{
    std::stringstream        stream(multiLineString);
    std::vector<std::string> res;
    while (1)
    {
        std::string line;
        std::getline(stream, line);
        if (!stream.good())
            break;
        line = Utils::trimString(line, "\r");
        res.push_back(line);
    }
    return res;
}
//-----------------------------------------------------------------------------
// Loads a file into a string and returns it
string readTextFileIntoString(const char* logTag, const string& pathAndFilename)
{
    fstream shaderFile(pathAndFilename.c_str(), std::ios::in);

    if (!shaderFile.is_open())
    {
        log(logTag,
            "File open failed in readTextFileIntoString: %s",
            pathAndFilename.c_str());
        exit(1);
    }

    std::stringstream buffer;
    buffer << shaderFile.rdbuf();
    return buffer.str();
}
//-----------------------------------------------------------------------------
// Writes a string into a text file
void writeStringIntoTextFile(const char*   logTag,
                             const string& stringToWrite,
                             const string& pathAndFilename)
{
    std::ofstream file(pathAndFilename);
    file << stringToWrite;
    if (file.bad())
        log(logTag,
            "Writing file failed in writeStringIntoTextFile: %s",
            pathAndFilename.c_str());
    file.close();
}
//-----------------------------------------------------------------------------
// deletes non-filename characters: /\|?%*:"<>'
string replaceNonFilenameChars(string src, const char replaceChar)
{
    std::replace(src.begin(), src.end(), '/', replaceChar);
    std::replace(src.begin(), src.end(), '\\', replaceChar);
    std::replace(src.begin(), src.end(), '|', replaceChar);
    std::replace(src.begin(), src.end(), '?', replaceChar);
    std::replace(src.begin(), src.end(), '%', replaceChar);
    std::replace(src.begin(), src.end(), '*', replaceChar);
    std::replace(src.begin(), src.end(), ':', replaceChar);
    std::replace(src.begin(), src.end(), '"', replaceChar);
    return src;
}
//-----------------------------------------------------------------------------
// Returns local time as string like "Wed Feb 13 15:46:11 2019"
string getLocalTimeString()
{
    time_t tm = 0;
    time(&tm);
    struct tm* t2 = localtime(&tm);
    char       buf[1024];
    strftime(buf, sizeof(buf), "%c", t2);
    return string(buf);
}
//-----------------------------------------------------------------------------
// Returns local time as string like "13.02.19-15:46"
string getDateTime1String()
{
    time_t tm = 0;
    time(&tm);
    struct tm* t = localtime(&tm);

    static char shortTime[50];
    snprintf(shortTime,
             sizeof(shortTime),
             "%.2d.%.2d.%.2d-%.2d:%.2d",
             t->tm_mday,
             t->tm_mon + 1,
             t->tm_year - 100,
             t->tm_hour,
             t->tm_min);

    return string(shortTime);
}
//-----------------------------------------------------------------------------
// Returns local time as string like "20190213-154611"
string getDateTime2String()
{
    time_t tm = 0;
    time(&tm);
    struct tm* t = localtime(&tm);

    static char shortTime[50];
    snprintf(shortTime,
             sizeof(shortTime),
             "%.4d%.2d%.2d-%.2d%.2d%.2d",
             1900 + t->tm_year,
             t->tm_mon + 1,
             t->tm_mday,
             t->tm_hour,
             t->tm_min,
             t->tm_sec);

    return string(shortTime);
}
//-----------------------------------------------------------------------------
// Returns the hostname from boost asio
string getHostName()
{
#ifndef __EMSCRIPTEN__
    return asio::ip::host_name();
#else
    return "0.0.0.0";
#endif
}
//-----------------------------------------------------------------------------
// Returns a formatted string as sprintf
string formatString(string fmt_str, ...)
{
    // Reserve two times as much as the length of the fmt_str
    int final_n = 0;
    int n       = ((int)fmt_str.size()) * 2;

    string                  str;
    std::unique_ptr<char[]> formatted;
    va_list                 ap;
    while (true)
    {
        formatted.reset(new char[n]);
        strcpy(&formatted[0], fmt_str.c_str());
        va_start(ap, fmt_str);
        final_n = vsnprintf(&formatted[0], (unsigned long)n, fmt_str.c_str(), ap);
        va_end(ap);
        if (final_n < 0 || final_n >= n)
            n += abs(final_n - n + 1);
        else
            break;
    }
    return string(formatted.get());
}
//-----------------------------------------------------------------------------
// Returns true if container contains the search string
bool containsString(const string& container, const string& search)
{
    return (container.find(search) != string::npos);
}
//-----------------------------------------------------------------------------
// Return true if the container string starts with the startStr
bool startsWithString(const string& container, const string& startStr)
{
    return container.find(startStr) == 0;
}
//-----------------------------------------------------------------------------
// Return true if the container string ends with the endStr
bool endsWithString(const string& container, const string& endStr)
{
    if (container.length() >= endStr.length())
        return (0 == container.compare(container.length() - endStr.length(),
                                       endStr.length(),
                                       endStr));
    else
        return false;
}
//-----------------------------------------------------------------------------
// Returns inputDir with unified forward slashes
string unifySlashes(const string& inputDir, bool withTrailingSlash)
{
    string copy = inputDir;
    string curr;
    string delimiter = "\\";
    size_t pos       = 0;
    string token;
    while ((pos = copy.find(delimiter)) != string::npos)
    {
        token = copy.substr(0, pos);
        copy.erase(0, pos + delimiter.length());
        curr.append(token);
        curr.append("/");
    }

    curr.append(copy);

    if (withTrailingSlash && !curr.empty() && curr.back() != '/')
        curr.append("/");

    return curr;
}
//-----------------------------------------------------------------------------
// Returns the path w. '\\' of path-filename string
string getPath(const string& pathFilename)
{
    size_t i1 = pathFilename.rfind('\\', pathFilename.length());
    size_t i2 = pathFilename.rfind('/', pathFilename.length());
    if ((i1 != string::npos && i2 == string::npos) ||
        (i1 != string::npos && i1 > i2))
    {
        return (pathFilename.substr(0, i1 + 1));
    }

    if ((i2 != string::npos && i1 == string::npos) ||
        (i2 != string::npos && i2 > i1))
    {
        return (pathFilename.substr(0, i2 + 1));
    }
    return pathFilename;
}
//-----------------------------------------------------------------------------
// Returns true if content of file could be put in a vector of strings
bool getFileContent(const string&   fileName,
                    vector<string>& vecOfStrings)
{

    // Open the File
    std::ifstream in(fileName.c_str());

    // Check if object is valid
    if (!in)
    {
        std::cerr << "Cannot open the File : " << fileName << std::endl;
        return false;
    }

    // Read the next line from File untill it reaches the end.
    std::string str;
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (!str.empty())
            vecOfStrings.push_back(str);
    }

    // Close The File
    in.close();
    return true;
}
//-----------------------------------------------------------------------------
// Naturally compares two strings (used for filename sorting)
/*! String comparison as most filesystem do it.
Source: https://www.o-rho.com/naturalsort

std::sort   compareNatural
---------   --------------
1.txt       1.txt
10.txt      1_t.txt
1_t.txt     10.txt
20          20
20.txt      20.txt
ABc         ABc
aBCd        aBCd
aBCd(01)    aBCd(1)
aBCd(1)     aBCd(01)
aBCd(12)    aBCd(2)
aBCd(2)     aBCd(12)
aBc         aBc
aBcd        aBcd
aaA         aaA
aaa         aaa
z10.txt     z2.txt
z100.txt    z10.txt
z2.txt      z100.txt
 */
bool compareNatural(const string& a, const string& b)
{
    const char*          p1         = a.c_str();
    const char*          p2         = b.c_str();
    const unsigned short st_scan    = 0;
    const unsigned short st_alpha   = 1;
    const unsigned short st_numeric = 2;
    unsigned short       state      = st_scan;
    const char*          numstart1  = nullptr;
    const char*          numstart2  = nullptr;
    const char*          numend1    = nullptr;
    const char*          numend2    = nullptr;
    unsigned long        sz1        = 0;
    unsigned long        sz2        = 0;

    while (*p1 && *p2)
    {
        switch (state)
        {
            case st_scan:
                if (!isdigit(*p1) && !isdigit(*p2))
                {
                    state = st_alpha;
                    if (*p1 == *p2)
                    {
                        p1++;
                        p2++;
                    }
                    else
                        return *p1 < *p2;
                }
                else if (isdigit(*p1) && !isdigit(*p2))
                    return true;
                else if (!isdigit(*p1) && isdigit(*p2))
                    return false;
                else
                {
                    state = st_numeric;
                    if (sz1 == 0)
                        while (*p1 == '0')
                        {
                            p1++;
                            sz1++;
                        }
                    else
                        while (*p1 == '0') p1++;
                    if (sz2 == 0)
                        while (*p2 == '0')
                        {
                            p2++;
                            sz2++;
                        }
                    else
                        while (*p2 == '0') p2++;
                    if (sz1 == sz2)
                    {
                        sz1 = 0;
                        sz2 = 0;
                    }
                    if (!isdigit(*p1)) p1--;
                    if (!isdigit(*p2)) p2--;
                    numstart1 = p1;
                    numstart2 = p2;
                    numend1   = numstart1;
                    numend2   = numstart2;
                }
                break;
            case st_alpha:
                if (!isdigit(*p1) && !isdigit(*p2))
                {
                    if (*p1 == *p2)
                    {
                        p1++;
                        p2++;
                    }
                    else
                        return *p1 < *p2;
                }
                else
                    state = st_scan;
                break;
            case st_numeric:
                while (isdigit(*p1)) numend1 = p1++;
                while (isdigit(*p2)) numend2 = p2++;
                if (numend1 - numstart1 == numend2 - numstart2 &&
                    !strncmp(numstart1, numstart2, numend2 - numstart2 + 1))
                    state = st_scan;
                else
                {
                    if (numend1 - numstart1 != numend2 - numstart2)
                        return numend1 - numstart1 < numend2 - numstart2;
                    while (*numstart1 && *numstart2)
                    {
                        if (*numstart1 != *numstart2) return *numstart1 < *numstart2;
                        numstart1++;
                        numstart2++;
                    }
                }
                break;
            default: break;
        }
    }
    if (sz1 < sz2) return true;
    if (sz1 > sz2) return false;
    if (*p1 == 0 && *p2 != 0) return true;
    if (*p1 != 0 && *p2 == 0) return false;
    return false;
}
//-----------------------------------------------------------------------------

/////////////////////////////
// File Handling Functions //
/////////////////////////////

//-----------------------------------------------------------------------------
// Returns the filename of path-filename string
string getFileName(const string& pathFilename)
{
    size_t i1 = pathFilename.rfind('\\', pathFilename.length());
    size_t i2 = pathFilename.rfind('/', pathFilename.length());
    int    i  = -1;

    if (i1 != string::npos && i2 != string::npos)
        i = (int)std::max(i1, i2);
    else if (i1 != string::npos)
        i = (int)i1;
    else if (i2 != string::npos)
        i = (int)i2;

    return pathFilename.substr(i + 1, pathFilename.length() - i);
}

//-----------------------------------------------------------------------------
// Returns the path of a path-filename combo
string getDirName(const string& pathFilename)
{
    size_t i1 = pathFilename.rfind('\\', pathFilename.length());
    size_t i2 = pathFilename.rfind('/', pathFilename.length());
    int    i  = -1;

    if (i1 != string::npos && i2 != string::npos)
        i = (int)std::max(i1, i2);
    else if (i1 != string::npos)
        i = (int)i1;
    else if (i2 != string::npos)
        i = (int)i2;

    return pathFilename.substr(0, i + 1);
}

//-----------------------------------------------------------------------------
// Returns the filename without extension
string getFileNameWOExt(const string& pathFilename)
{
    string filename = getFileName(pathFilename);
    size_t i        = filename.rfind('.', filename.length());
    if (i != string::npos)
    {
        return (filename.substr(0, i));
    }

    return (filename);
}
//-----------------------------------------------------------------------------
// Returns the file extension without dot in lower case
string getFileExt(const string& filename)
{
    size_t i = filename.rfind('.', filename.length());
    if (i != string::npos)
        return toLowerString(filename.substr(i + 1, filename.length() - i));
    return ("");
}
//-----------------------------------------------------------------------------
// Returns a vector of unsorted directory names with path in dir
vector<string> getDirNamesInDir(const string& dirName, bool fullPath)
{
    vector<string> filePathNames;

#if defined(USE_STD_FILESYSTEM)
    if (fs::exists(dirName) && fs::is_directory(dirName))
    {
        for (const auto& entry : fs::directory_iterator(dirName))
        {
            auto filename = entry.path().filename();
            if (fs::is_directory(entry.status()))
            {
                if (fullPath)
                    filePathNames.push_back(dirName + "/" + filename.u8string());
                else
                    filePathNames.push_back(filename.u8string());
            }
        }
    }
#else
    DIR* dir = opendir(dirName.c_str());

    if (dir)
    {
        struct dirent* dirContent = nullptr;
        int            i          = 0;

        while ((dirContent = readdir(dir)) != nullptr)
        {
            i++;
            string name(dirContent->d_name);

            if (name != "." && name != "..")
            {
                struct stat path_stat
                {
                };
                stat((dirName + name).c_str(), &path_stat);
                if (!S_ISREG(path_stat.st_mode))
                {
                    if (fullPath)
                        filePathNames.push_back(dirName + "/" + name);
                    else
                        filePathNames.push_back(name);
                }
            }
        }
        closedir(dir);
    }
#endif

    return filePathNames;
}
//-----------------------------------------------------------------------------
// Returns a vector of unsorted names (files and directories) with path in dir
vector<string> getAllNamesInDir(const string& dirName, bool fullPath)
{
    vector<string> filePathNames;

#if defined(USE_STD_FILESYSTEM)
    if (fs::exists(dirName) && fs::is_directory(dirName))
    {
        for (const auto& entry : fs::directory_iterator(dirName))
        {
            auto filename = entry.path().filename();
            if (fullPath)
                filePathNames.push_back(dirName + "/" + filename.u8string());
            else
                filePathNames.push_back(filename.u8string());
        }
    }
#else
#    if defined(TARGET_OS_IOS) && (TARGET_OS_IOS == 1)
    return Utils_iOS::getAllNamesInDir(dirName, fullPath);
#    else
    DIR* dir = opendir(dirName.c_str());

    if (dir)
    {
        struct dirent* dirContent = nullptr;
        int            i          = 0;

        while ((dirContent = readdir(dir)) != nullptr)
        {
            i++;
            string name(dirContent->d_name);
            if (name != "." && name != "..")
            {
                if (fullPath)
                    filePathNames.push_back(dirName + "/" + name);
                else
                    filePathNames.push_back(name);
            }
        }
        closedir(dir);
    }
#    endif
#endif

    return filePathNames;
}
//-----------------------------------------------------------------------------
// Returns a vector of unsorted filesnames with path in dir
vector<string> getFileNamesInDir(const string& dirName, bool fullPath)
{
    vector<string> filePathNames;

#if defined(USE_STD_FILESYSTEM)
    if (fs::exists(dirName) && fs::is_directory(dirName))
    {
        for (const auto& entry : fs::directory_iterator(dirName))
        {
            auto filename = entry.path().filename();
            if (fs::is_regular_file(entry.status()))
            {
                if (fullPath)
                    filePathNames.push_back(dirName + "/" + filename.u8string());
                else
                    filePathNames.push_back(filename.u8string());
            }
        }
    }
#else
    // todo: does this part also return directories? It should only return file names..
    DIR* dir = opendir(dirName.c_str());

    if (dir)
    {
        struct dirent* dirContent = nullptr;
        int            i          = 0;

        while ((dirContent = readdir(dir)) != nullptr)
        {
            i++;
            string name(dirContent->d_name);
            if (name != "." && name != "..")
            {
                struct stat path_stat
                {
                };
                stat((dirName + name).c_str(), &path_stat);
                if (S_ISREG(path_stat.st_mode))
                {
                    if (fullPath)
                        filePathNames.push_back(dirName + name);
                    else
                        filePathNames.push_back(name);
                }
            }
        }
        closedir(dir);
    }
#endif

    return filePathNames;
}
//-----------------------------------------------------------------------------
// Returns true if a directory exists.
bool dirExists(const string& path)
{
#if defined(__EMSCRIPTEN__)
    return true;
#elif defined(USE_STD_FILESYSTEM)
    return fs::exists(path) && fs::is_directory(path);
#else
    struct stat info
    {
    };
    if (stat(path.c_str(), &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR)
        return true;
    else
        return false;
#endif
}
//-----------------------------------------------------------------------------
// Creates a directory with given path
bool makeDir(const string& path)
{
#if defined(USE_STD_FILESYSTEM)
    return fs::create_directories(path);
#else
#    if defined(_WIN32)
    return _mkdir(path.c_str());
#    else
    int  failed = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    bool result = !failed;
    return result;
#    endif
#endif
}
//-----------------------------------------------------------------------------
// Creates a directory with given path recursively
bool makeDirRecurse(std::string path)
{
    std::string delimiter = "/";

    size_t      pos = 0;
    std::string token;

    std::string createdPath;

    while ((pos = path.find(delimiter)) != std::string::npos)
    {
        createdPath += path.substr(0, pos) + "/";

        if (!dirExists(createdPath))
        {
            if (!makeDir(createdPath))
            {
                return false;
            }
        }

        path.erase(0, pos + delimiter.length());
    }

    return true;
}
//-----------------------------------------------------------------------------
// Removes a directory with given path
void removeDir(const string& path)
{

#if defined(USE_STD_FILESYSTEM)
    fs::remove_all(path);
#else
#    if defined(_WIN32)
    int ret = _rmdir(path.c_str());
    if (ret != 0)
    {
        errno_t err;
        _get_errno(&err);
        log("Could not remove directory: %s\nErrno: %s\n", path.c_str(), strerror(errno));
    }
#    else
    rmdir(path.c_str());
#    endif
#endif
}
//-----------------------------------------------------------------------------
// Removes a file with given path
void removeFile(const string& path)
{
    if (fileExists(path))
    {
#if defined(USE_STD_FILESYSTEM)
        fs::remove(path);
#else
#    if defined(_WIN32)
        DeleteFileA(path.c_str());
#    else
        unlink(path.c_str());
#    endif

#endif
    }
    else
        log("Could not remove file : %s\nErrno: %s\n",
            path.c_str(),
            "file does not exist");
}
//-----------------------------------------------------------------------------
// Returns true if a file exists.
bool fileExists(const string& pathfilename)
{
#if defined(__EMSCRIPTEN__)
    return false;
#elif defined(USE_STD_FILESYSTEM)
    return fs::exists(pathfilename);
#else
    struct stat info
    {
    };
    return (stat(pathfilename.c_str(), &info) == 0) && ((info.st_mode & S_IFDIR) == 0);
#endif
}
//-----------------------------------------------------------------------------
// Returns the file size in bytes
unsigned int getFileSize(const string& pathfilename)
{
#if defined(USE_STD_FILESYSTEM)
    if (fs::exists(pathfilename))
        return (unsigned int)fs::file_size(pathfilename);
    else
        return 0;
#else
    struct stat st
    {
    };
    if (stat(pathfilename.c_str(), &st) != 0)
        return 0;
    return (unsigned int)st.st_size;
#endif
}
//-----------------------------------------------------------------------------
// Returns the file size in bytes
unsigned int getFileSize(std::ifstream& fs)
{
    fs.seekg(0, std::ios::beg);
    std::streampos begin = fs.tellg();
    fs.seekg(0, std::ios::end);
    std::streampos end = fs.tellg();
    fs.seekg(0, std::ios::beg);
    return (unsigned int)(end - begin);
}

//-----------------------------------------------------------------------------
// Returns the writable configuration directory with trailing forward slash
string getAppsWritableDir(string appName)
{
#if defined(_WIN32)
    string appData   = getenv("APPDATA");
    string configDir = appData + "/" + appName;
    replaceString(configDir, "\\", "/");
    if (!dirExists(configDir))
        makeDir(configDir.c_str());
    return configDir + "/";
#elif defined(__APPLE__)
    string home      = getenv("HOME");
    string appData   = home + "/Library/Application Support";
    string configDir = appData + "/" + appName;
    if (!dirExists(configDir))
        mkdir(configDir.c_str(), S_IRWXU);
    return configDir + "/";
#elif defined(ANDROID) || defined(ANDROID_NDK)
    // @todo Where is the app data path on Andoroid?
#elif defined(linux) || defined(__linux) || defined(__linux__)
    // @todo Where is the app data path on Linux?
    string home      = getenv("HOME");
    string configDir = home + "/." + appName;
    if (!dirExists(configDir))
        mkdir(configDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    return configDir + "/";
#elif defined(__EMSCRIPTEN__)
    return "?";
#else
#    error "No port to this OS"
#endif
}
//-----------------------------------------------------------------------------
// Returns the working directory with forward slashes inbetween and at the end
string getCurrentWorkingDir()
{
#if defined(_WIN32)
#    if defined(USE_STD_FILESYSTEM)
    return fs::current_path().u8string();
#    else
    int   size   = 256;
    char* buffer = (char*)malloc(size);
    if (_getcwd(buffer, size) == buffer)
    {
        string dir = buffer;
        replaceString(dir, "\\", "/");
        return dir + "/";
    }

    free(buffer);
    return "";
#    endif
#elif !defined(__EMSCRIPTEN__)
    size_t size   = 256;
    char*  buffer = (char*)malloc(size);
    if (getcwd(buffer, size) == buffer)
        return string(buffer) + "/";

    free(buffer);
    return "";
#else
    return "/";
#endif
}
//-----------------------------------------------------------------------------
// Deletes a file on the filesystem
bool deleteFile(string& pathfilename)
{
    if (fileExists(pathfilename))
        return remove(pathfilename.c_str()) != 0;
    return false;
}
//-----------------------------------------------------------------------------
// process all files and folders recursively naturally sorted
void loopFileSystemRec(const string&                                           path,
                       function<void(string path, string baseName, int depth)> processFile,
                       function<void(string path, string baseName, int depth)> processDir,
                       const int                                               depth)
{
    // be sure that the folder slashes are correct
    string folder = unifySlashes(path);

    if (dirExists(folder))
    {
        vector<string> unsortedNames = getAllNamesInDir(folder);

        processDir(getDirName(trimRightString(folder, "/")),
                   getFileName(trimRightString(folder, "/")),
                   depth);
        sort(unsortedNames.begin(), unsortedNames.end(), Utils::compareNatural);

        for (const auto& fileOrFolder : unsortedNames)
        {
            if (dirExists(fileOrFolder))
                loopFileSystemRec(fileOrFolder, processFile, processDir, depth + 1);
            else
                processFile(folder, getFileName(fileOrFolder), depth);
        }
    }
    else
    {
        processFile(getDirName(trimRightString(path, "/")),
                    getFileName(trimRightString(path, "/")),
                    depth);
    }
}

//-----------------------------------------------------------------------------
// Dumps all files and folders on stdout recursively naturally sorted
void dumpFileSystemRec(const char* logtag, const string& folderPath)
{
    const char* tab = "    ";

    loopFileSystemRec(
      folderPath,
      [logtag, tab](string path, string baseName, int depth) -> void
      {
          string indent;
          for (int d = 0; d < depth; ++d)
              indent += tab;
          string indentFolderName = indent + baseName;
          Utils::log(logtag, "%s", indentFolderName.c_str());
      },
      [logtag, tab](string path, string baseName, int depth) -> void
      {
          string indent;
          for (int d = 0; d < depth; ++d)
              indent += tab;
          string indentFolderName = indent + "[" + baseName + "]";
          Utils::log(logtag, "%s", indentFolderName.c_str());
      });
}
//-----------------------------------------------------------------------------
// findFile return the full path with filename
/* Unfortunatelly the relative folder structure on different OS are not identical.
 * This function allows to search on for a file on different paths.
 */
string findFile(const string& filename, const vector<string>& pathsToCheck)
{
    if (Utils::fileExists(filename))
        return filename;

    // Check file existence
    for (const auto& path : pathsToCheck)
    {
        string pathPlusFilename = Utils::unifySlashes(path) + filename;
        if (Utils::fileExists(pathPlusFilename))
            return pathPlusFilename;
    }
    return "";
}
//----------------------------------------------------------------------------

///////////////////////
// Logging Functions //
///////////////////////
//-----------------------------------------------------------------------------
void initFileLog(const string& logDir, bool forceFlush)
{
    fileLog = std::make_unique<FileLog>(logDir, forceFlush);
}
//-----------------------------------------------------------------------------
// logs a formatted string platform independently
void log(const char* tag, const char* format, ...)
{
    char log[4096];

    va_list argptr;
    va_start(argptr, format);
    vsnprintf(log, sizeof(log), format, argptr);
    va_end(argptr);

    char msg[4096];
    strcpy(msg, tag);
    strcat(msg, ": ");
    strcat(msg, log);
    strcat(msg, "\n");

    if (fileLog)
        fileLog->post(msg);

    if (customLog)
        customLog->post(msg);

#if defined(ANDROID) || defined(ANDROID_NDK)
    __android_log_print(ANDROID_LOG_INFO, tag, msg);
#else
    std::cout << msg << std::flush;
#endif
}
//-----------------------------------------------------------------------------
// Terminates the application with a message. No leak checking.
void exitMsg(const char* tag,
             const char* msg,
             const int   line,
             const char* file)
{
#if defined(ANDROID) || defined(ANDROID_NDK)
    __android_log_print(ANDROID_LOG_FATAL,
                        tag,
                        "Exit %s at line %d in %s\n",
                        msg,
                        line,
                        file);
#elif defined(__EMSCRIPTEN__)
    std::string message = "SLProject error";
    alert(message.c_str());
#else
    log(tag,
        "Exit %s at line %d in %s\n",
        msg,
        line,
        file);
#endif

    exit(-1);
}
//-----------------------------------------------------------------------------
// Warn message output
void warnMsg(const char* tag,
             const char* msg,
             const int   line,
             const char* file)
{
#if defined(ANDROID) || defined(ANDROID_NDK)
    __android_log_print(ANDROID_LOG_WARN,
                        tag,
                        "Warning: %s at line %d in %s\n",
                        msg,
                        line,
                        file);
#else
    log(tag,
        "Warning %s at line %d in %s\n",
        msg,
        line,
        file);
#endif
}
//-----------------------------------------------------------------------------
// Error message output (same as warn but with another tag for android)
void errorMsg(const char* tag,
              const char* msg,
              const int   line,
              const char* file)
{
#if defined(ANDROID) || defined(ANDROID_NDK)
    __android_log_print(ANDROID_LOG_ERROR,
                        tag,
                        "Error: %s at line %d in %s\n",
                        msg,
                        line,
                        file);
#else
    log(tag,
        "Error %s at line %d in %s\n",
        msg,
        line,
        file);
#endif
}
//-----------------------------------------------------------------------------
// Shows the a spinner icon message. So far only used with emscripten
void showSpinnerMsg(string msg)
{
#ifdef __EMSCRIPTEN__
    // clang-format off
    MAIN_THREAD_EM_ASM({
        let resource = UTF8ToString($0);
        document.querySelector("#loading-text").innerHTML = resource;

        if (globalThis.hideTimer === null) {
            document.querySelector("#loading-overlay").classList.add("visible");
        } else {
            clearTimeout(globalThis.hideTimer);
        }
    }, msg.c_str());
    // clang-format on
#endif
}
//-----------------------------------------------------------------------------
// Hides the previous spinner icon message. So far only used with emscripten
void hideSpinnerMsg()
{
#ifdef __EMSCRIPTEN__
    // clang-format off
    MAIN_THREAD_EM_ASM({
        globalThis.hideTimer = setTimeout(function () {
              globalThis.hideTimer = null;
              document.querySelector("#loading-overlay").classList.remove("visible");
          }, 500);
    });
    // clang-format on
#endif
}
//-----------------------------------------------------------------------------
// Returns in release config the max. NO. of threads otherwise 1
unsigned int maxThreads()
{
#if defined(DEBUG) || defined(_DEBUG)
    return 1;
#else
    return std::max(std::thread::hardware_concurrency(), 1U);
#endif
}
//-----------------------------------------------------------------------------

////////////////////
// Math Utilities //
////////////////////

//-----------------------------------------------------------------------------
// Greatest common divisor of two integer numbers (ggT = grösster gemeinsame Teiler)
int gcd(int a, int b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}
//-----------------------------------------------------------------------------
// Lowest common multiple (kgV = kleinstes gemeinsames Vielfache)
int lcm(int a, int b)
{
    return (a * b) / Utils::gcd(a, b);
}
//-----------------------------------------------------------------------------
// Returns the closest power of 2 to a passed number.
unsigned closestPowerOf2(unsigned num)
{
    unsigned nextPow2 = 1;
    if (num <= 0) return 1;

    while (nextPow2 <= num)
        nextPow2 <<= 1;
    unsigned prevPow2 = nextPow2 >> 1;

    if (num - prevPow2 < nextPow2 - num)
        return prevPow2;
    else
        return nextPow2;
}
//-----------------------------------------------------------------------------
// Returns the next power of 2 to a passed number.
unsigned nextPowerOf2(unsigned num)
{
    unsigned nextPow2 = 1;
    if (num == 0) return 1;

    while (nextPow2 <= num)
        nextPow2 <<= 1;
    return nextPow2;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// ComputerInfos
//-----------------------------------------------------------------------------
std::string ComputerInfos::user  = "USER?";
std::string ComputerInfos::name  = "NAME?";
std::string ComputerInfos::brand = "BRAND?";
std::string ComputerInfos::model = "MODEL?";
std::string ComputerInfos::os    = "OS?";
std::string ComputerInfos::osVer = "OSVER?";
std::string ComputerInfos::arch  = "ARCH?";
std::string ComputerInfos::id    = "ID?";

//-----------------------------------------------------------------------------
std::string ComputerInfos::get()
{
#if defined(_WIN32) //..................................................

    // Computer user name
    const char* envvar = std::getenv("USER");
    user               = envvar ? string(envvar) : "USER?";
    if (user == "USER?")
    {
        const char* envvar = std::getenv("USERNAME");
        user               = envvar ? string(envvar) : "USER?";
    }
    name = Utils::getHostName();

    // Get architecture
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    switch (siSysInfo.wProcessorArchitecture)
    {
        case PROCESSOR_ARCHITECTURE_AMD64: arch = "x64"; break;
        case PROCESSOR_ARCHITECTURE_ARM: arch = "ARM"; break;
        case 12: arch = "ARM64"; break; // PROCESSOR_ARCHITECTURE_ARM64
        case PROCESSOR_ARCHITECTURE_IA64: arch = "IA64"; break;
        case PROCESSOR_ARCHITECTURE_INTEL: arch = "x86"; break;
        default: arch = "???";
    }

    // Windows OS version
    OSVERSIONINFO osInfo;
    ZeroMemory(&osInfo, sizeof(OSVERSIONINFO));
    osInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
    GetVersionEx(&osInfo);
    char osVersion[50];
    sprintf(osVersion, "%u.%u", osInfo.dwMajorVersion, osInfo.dwMinorVersion);
    osVer = string(osVersion);

    brand = "BRAND?";
    model = "MODEL?";
    os    = "Windows";

#elif defined(__APPLE__)
#    if defined(TARGET_OS_IOS) && (TARGET_OS_IOS == 1)
    // Model and architecture are retrieved before in iOS under Objective C
    brand              = "Apple";
    os                 = "iOS";
    const char* envvar = std::getenv("USER");
    user               = envvar ? string(envvar) : "USER?";
    if (user == "USER?")
    {
        const char* envvar = std::getenv("USERNAME");
        user               = envvar ? string(envvar) : "USER?";
    }
    name = Utils::getHostName();
#    else
    // Computer user name
    const char* envvar = std::getenv("USER");
    user               = envvar ? string(envvar) : "USER?";

    if (user == "USER?")
    {
        const char* envvarUN = std::getenv("USERNAME");
        user                 = envvarUN ? string(envvarUN) : "USER?";
    }

    name  = Utils::getHostName();
    brand = "Apple";
    os    = "MacOS";

    // Get MacOS version
    // SInt32 majorV, minorV, bugfixV;
    // Gestalt(gestaltSystemVersionMajor, &majorV);
    // Gestalt(gestaltSystemVersionMinor, &minorV);
    // Gestalt(gestaltSystemVersionBugFix, &bugfixV);
    // char osVer[50];
    // sprintf(osVer, "%d.%d.%d", majorV, minorV, bugfixV);
    // osVer = string(osVer);

    // Get model
    // size_t len = 0;
    // sysctlbyname("hw.model", nullptr, &len, nullptr, 0);
    // char model[255];
    // sysctlbyname("hw.model", model, &len, nullptr, 0);
    // model = model;
#    endif

#elif defined(ANDROID)                                         //................................................

    os = "Android";

    /*
    "ro.build.version.release"     // * The user-visible version string. E.g., "1.0" or "3.4b5".
    "ro.build.version.incremental" // The internal value used by the underlying source control to represent this build.
    "ro.build.version.codename"    // The current development codename, or the string "REL" if this is a release build.
    "ro.build.version.sdk"         // The user-visible SDK version of the framework.

    "ro.product.model"             // * The end-user-visible name for the end product..
    "ro.product.manufacturer"      // The manufacturer of the product/hardware.
    "ro.product.board"             // The name of the underlying board, like "goldfish".
    "ro.product.brand"             // The brand (e.g., carrier) the software is customized for, if any.
    "ro.product.device"            // The name of the industrial design.
    "ro.product.name"              // The name of the overall product.
    "ro.hardware"                  // The name of the hardware (from the kernel command line or /proc).
    "ro.product.cpu.abi"           // The name of the instruction set (CPU type + ABI convention) of native code.
    "ro.product.cpu.abi2"          // The name of the second instruction set (CPU type + ABI convention) of native code.

    "ro.build.display.id"          // * A build ID string meant for displaying to the user.
    "ro.build.host"
    "ro.build.user"
    "ro.build.id"                  // Either a changelist number, or a label like "M4-rc20".
    "ro.build.type"                // The type of build, like "user" or "eng".
    "ro.build.tags"                // Comma-separated tags describing the build, like "unsigned,debug".
    */

    int len;

    char hostC[PROP_VALUE_MAX];
    len  = __system_property_get("ro.build.host", hostC);
    name = hostC ? string(hostC) : "NAME?";

    char userC[PROP_VALUE_MAX];
    len  = __system_property_get("ro.build.user", userC);
    user = userC ? string(userC) : "USER?";

    char brandC[PROP_VALUE_MAX];
    len   = __system_property_get("ro.product.brand", brandC);
    brand = string(brandC);

    char modelC[PROP_VALUE_MAX];
    len   = __system_property_get("ro.product.model", modelC);
    model = string(modelC);

    char osVerC[PROP_VALUE_MAX];
    len   = __system_property_get("ro.build.version.release", osVerC);
    osVer = string(osVerC);

    char archC[PROP_VALUE_MAX];
    len  = __system_property_get("ro.product.cpu.abi", archC);
    arch = string(archC);

#elif defined(linux) || defined(__linux) || defined(__linux__) //..................................................

    os    = "Linux";
    user  = "USER?";
    name  = Utils::getHostName();
    brand = "BRAND?";
    model = "MODEL?";
    osVer = "OSVER?";
    arch  = "ARCH?";
#endif

    // build a unique as possible ID string that can be used in a filename
    id = user + "-" + name + "-" + model;
    if (model.find("SM-") != string::npos)
        // Don't use computerName on Samsung phones. It's not constant!
        id = user + "-" + model;
    else
        id = user + "-" + name + "-" + model;
    id = Utils::replaceNonFilenameChars(id);
    std::replace(id.begin(), id.end(), '_', '-');
    return id;
}
//-----------------------------------------------------------------------------
}
