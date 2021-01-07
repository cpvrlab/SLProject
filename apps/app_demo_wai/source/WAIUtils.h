#ifndef WAI_UTILS_H
#define WAI_UTILS_H

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <iomanip>
#include <sstream>

//#include <SL.h>
//#include <SLMath.h>
#include <dirent.h>

//-----------------------------------------------------------------------------
//! SLUtils provides static utility functions std::string handling
class WAIUtils
{
    public:
    //! SLUtils::toString returns a std::string from a float with max. one trailing zero
    static std::string toString(float f, int roundedDecimals = 1)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(roundedDecimals) << f;
        std::string num = ss.str();
        if (num == "-0.0") num = "0.0";
        return num;
    }

    //! SLUtils::toString returns a std::string from a double with max. one trailing zero
    static std::string toString(double d, int roundedDecimals = 1)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(roundedDecimals) << d;
        std::string num = ss.str();
        if (num == "-0.0") num = "0.0";
        return num;
    }

    //! SLUtils::toLower returns a std::string in lower case
    static std::string toLower(std::string s)
    {
        std::string cpy(s);
        transform(cpy.begin(), cpy.end(), cpy.begin(), ::tolower);
        return cpy;
    }

    //! SLUtils::toUpper returns a std::string in upper case
    static std::string toUpper(std::string s)
    {
        std::string cpy(s);
        transform(cpy.begin(), cpy.end(), cpy.begin(), ::toupper);
        return cpy;
    }

    //! SLUtils::getPath returns the path w. '\\' of path-filename std::string
    static std::string getPath(const std::string& pathFilename)
    {
        size_t i1, i2;
        i1 = pathFilename.rfind('\\', pathFilename.length());
        i2 = pathFilename.rfind('/', pathFilename.length());
        if ((i1 != std::string::npos && i2 == std::string::npos) ||
            (i1 != std::string::npos && i1 > i2))
        {
            return (pathFilename.substr(0, i1 + 1));
        }

        if ((i2 != std::string::npos && i1 == std::string::npos) ||
            (i2 != std::string::npos && i2 > i1))
        {
            return (pathFilename.substr(0, i2 + 1));
        }
        return pathFilename;
    }

    //! SLUtils::getFileName returns the filename of path-filename std::string
    static std::string getFileName(const std::string& pathFilename)
    {
        size_t i = 0, i1, i2;
        i1       = pathFilename.rfind('\\', pathFilename.length());
        i2       = pathFilename.rfind('/', pathFilename.length());

        if (i1 != std::string::npos && i2 != std::string::npos)
            i = std::max(i1, i2);
        else if (i1 != std::string::npos)
            i = i1;
        else if (i2 != std::string::npos)
            i = i2;

        return pathFilename.substr(i + 1, pathFilename.length() - i);
    }

    //! SLUtils::getFileNameWOExt returns the filename without extension
    static std::string getFileNameWOExt(const std::string& pathFilename)
    {
        std::string filename = getFileName(pathFilename);
        size_t      i;
        i = filename.rfind('.', filename.length());
        if (i != std::string::npos)
        {
            return (filename.substr(0, i));
        }

        return (filename);
    }

    //! SLUtils::getFileExt returns the file extension without dot in lower case
    static std::string getFileExt(std::string filename)
    {
        size_t i;
        i = filename.rfind('.', filename.length());
        if (i != std::string::npos)
            return toLower(filename.substr(i + 1, filename.length() - i));
        return ("");
    }

    //! SLUtils::getFileNamesinDir returns a vector of storted filesname with path within a directory
    static std::vector<std::string> getFileNamesInDir(const std::string dirName)
    {
        std::vector<std::string> fileNames;
        DIR*                     dir;
        dir = opendir(dirName.c_str());

        if (dir)
        {
            struct dirent* dirContent;
            int            i = 0;

            while ((dirContent = readdir(dir)) != nullptr)
            {
                i++;
                std::string name(dirContent->d_name);
                if (name != "." && name != "..")
                    fileNames.push_back(dirName + "/" + name);
            }
            closedir(dir);
        }
        return fileNames;
    }

    //! SLUtils::trims a std::string at the end
    static std::string trim(std::string& s, const std::string& drop = " ")
    {
        std::string r = s.erase(s.find_last_not_of(drop) + 1);
        return r.erase(0, r.find_first_not_of(drop));
    }

    //! SLUtils::splits an input std::string at a delimeter character into a std::string vector
    static void split(const std::string& s, char delimiter, std::vector<std::string>& splits)
    {
        std::string::size_type i = 0;
        std::string::size_type j = s.find(delimiter);

        while (j != std::string::npos)
        {
            splits.push_back(s.substr(i, j - i));
            i = ++j;
            j = s.find(delimiter, j);
            if (j == std::string::npos)
                splits.push_back(s.substr(i, s.length()));
        }
    }

    //! Replaces in the subject std::string the search std::string by the replace std::string
    static void replaceString(std::string&       source,
                              const std::string& from,
                              const std::string& to)
    {
        // Code from: http://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
        std::string newString;
        newString.reserve(source.length()); // avoids a few memory allocations

        std::string::size_type lastPos = 0;
        std::string::size_type findPos;

        while (std::string::npos != (findPos = source.find(from, lastPos)))
        {
            newString.append(source, lastPos, findPos - lastPos);
            newString += to;
            lastPos = findPos + from.length();
        }

        // Care for the rest after last occurrence
        newString += source.substr(lastPos);
        source.swap(newString);
    }

    //! Returns local time as std::string
    static std::string getLocalTimeString()
    {
        time_t tm;
        time(&tm);
        struct tm* t2 = localtime(&tm);
        char       buf[1024];
        strftime(buf, sizeof(buf), "%c", t2);
        return std::string(buf);
    }

    //! Returns a formatted std::string as sprintf
    static std::string formatString(const std::string fmt_str, ...)
    {
        // Reserve two times as much as the length of the fmt_str
        int                     final_n, n = ((int)fmt_str.size()) * 2;
        std::string             str;
        std::unique_ptr<char[]> formatted;
        va_list                 ap;
        while (1)
        {
            formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
            strcpy(&formatted[0], fmt_str.c_str());
            va_start(ap, fmt_str);
            final_n = vsnprintf(&formatted[0], (unsigned long)n, fmt_str.c_str(), ap);
            va_end(ap);
            if (final_n < 0 || final_n >= n)
                n += abs(final_n - n + 1);
            else
                break;
        }
        return std::string(formatted.get());
    }

    //! contains returns true if container contains the search std::string
    static bool contains(const std::string container, const std::string search)
    {
        return (container.find(search) != std::string::npos);
    }

    //! Check, that slashes in directory std::string are definded forward with an additional slath at the end, e.g.: "dirA/dirB/"
    static std::string unifySlashes(const std::string& inputDir)
    {
        std::string copy = inputDir;
        std::string curr;
        std::string delimiter = "\\";
        size_t      pos       = 0;
        std::string token;
        while ((pos = copy.find(delimiter)) != std::string::npos)
        {
            token = copy.substr(0, pos);
            copy.erase(0, pos + delimiter.length());
            curr.append(token);
            curr.append("/");
        }

        curr.append(copy);
        if (curr.size() && curr.back() != '/')
            curr.append("/");

        return curr;
    }
};
//-----------------------------------------------------------------------------
#endif
