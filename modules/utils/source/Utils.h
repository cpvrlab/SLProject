//#############################################################################
//  File:      Utils.h
//  Authors:   Marcus Hudritsch
//  Date:      May 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CPLVRLAB_UTILS_H
#define CPLVRLAB_UTILS_H

#include <string>
#include <vector>
#include <cfloat>
#include <memory>
#include <FileLog.h>
#include <CustomLog.h>
#include <functional>

using std::function;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

// class FileLog;
//-----------------------------------------------------------------------------
//! Utils provides utilities for string & file handling, logging and math functions
/*!
 Function are grouped into sections:
 - String Handling Functions
 - File Handling Functions
 - Logging Functions
 - Math Constants and Functions
*/
namespace Utils
{
///////////////////////////////
// String Handling Functions //
///////////////////////////////

//! Returns a string from a float with max. one trailing zero
string toString(float f, int roundedDecimals = 1);

//! Returns a string from a double with max. one trailing zero
string toString(double d, int roundedDecimals = 1);

//! Returns a string in lower case
string toLowerString(string s);

//! Returns a string in upper case
string toUpperString(string s);

//! Trims a string at both end
string trimString(const string& s, const string& drop = " ");

//! trims a string at the right end
string trimRightString(const string& s, const string& drop);

//! trims a string at the left end
string trimLeftString(const string& s, const string& drop);

//! Splits an input string at a delimiter character into a string vector
void splitString(const string& s, char delimiter, vector<string>& splits);

//! Replaces in the source string the from string by the to string
void replaceString(string& source, const string& from, const string& to);

//! Returns a vector of string one per line of a multiline string
vector<string> getStringLines(const string& multiLineString);

//! Reads a text file into a string and returns it
string readTextFileIntoString(const char*   logTag,
                              const string& pathAndFilename);

//! Writes a string into a text file
void writeStringIntoTextFile(const char*   logTag,
                             const string& stringToWrite,
                             const string& pathAndFilename);

//! replaces non-filename characters: /\|?%*:"<>'
string replaceNonFilenameChars(string source, char replaceChar = '-');

//! Returns local time as string like "Wed Feb 13 15:46:11 2019"
string getLocalTimeString();

//! Returns local time as string like "13.02.19-15:46"
string getDateTime1String();

//! Returns local time as string like "20190213-154611"
string getDateTime2String();

//! Returns the computer name
string getHostName();

//! Returns a formatted string as sprintf
string formatString(string fmt_str, ...);

//! Returns true if container contains the search string
bool containsString(const string& container, const string& search);

//! Return true if the container string starts with the startStr
bool startsWithString(const string& container, const string& startStr);

//! Return true if the container string ends with the endStr
bool endsWithString(const string& container, const string& endStr);

//! Returns the inputDir string with unified forward slashes, e.g.: "dirA/dirB/"
string unifySlashes(const string& inputDir, bool withTrailingSlash = true);

//! Returns true if content of file could be put in a vector of strings
bool getFileContent(const string&   fileName,
                    vector<string>& vecOfStrings);

//! Naturally compares two strings (used for filename sorting)
bool compareNatural(const string& a, const string& b);

/////////////////////////////
// File Handling Functions //
/////////////////////////////

//! Returns the path w. '\\' of path-filename string
string getPath(const string& pathFilename);

//! Returns the filename of path-filename string
string getFileName(const string& pathFilename);

//! Returns the filename without extension
string getFileNameWOExt(const string& pathFilename);

//! Strip last component from file name
string getDirName(const string& pathFilename);

//! Returns the file extension without dot in lower case
string getFileExt(const string& filename);

//! Returns a vector directory names with path in dir
vector<string> getDirNamesInDir(const string& dirName, bool fullPath = true);

//! Returns a vector of sorted names (files and directories) with path in dir
vector<string> getAllNamesInDir(const string& dirName, bool fullPath = true);

//! Returns a vector of sorted filesnames in dirName
vector<string> getFileNamesInDir(const string& dirName, bool fullPath = true);

//! Returns true if a directory exists.
bool dirExists(const string& path);

//! Returns the file size in bytes
unsigned int getFileSize(const string& filename);
unsigned int getFileSize(std::ifstream& fs);

//! Creates a directory with given path
bool makeDir(const string& path);

//! Creates a directory with given path recursively
bool makeDirRecurse(string path);

//! RemoveDir deletes a directory with given path
void removeDir(const string& path);

//! RemoveFile deletes a file with given path
void removeFile(const string& path);

//! Returns true if a file exists.
bool fileExists(const string& pathfilename);

//! Returns the writable configuration directory
string getAppsWritableDir(string appName = "SLProject");

//! Returns the working directory
string getCurrentWorkingDir();

//! Deletes a file on the filesystem
bool deleteFile(string& pathfilename);

//! process all files and folders recursively naturally sorted
void loopFileSystemRec(const string&                                           path,
                       function<void(string path, string baseName, int depth)> processFile,
                       function<void(string path, string baseName, int depth)> processDir,
                       const int                                               depth = 0);

//! Dumps all folders and files recursovely
void dumpFileSystemRec(const char*   logtag,
                       const string& folderpath);

//! Tries to find a filename on various paths to check
string findFile(const string&         filename,
                const vector<string>& pathsToCheck);

///////////////////////
// Logging Functions //
///////////////////////

//! FileLog Instance for logging to logfile. If it is instantiated the logging methods
//! will also output into this file. Instantiate it with initFileLog function.
static std::unique_ptr<FileLog> fileLog;

//! Instantiates FileLog instance
void initFileLog(const std::string& logDir, bool forceFlush);

//! custom log instance, e.g. log to a ui log window
extern std::unique_ptr<CustomLog> customLog;

//! logs a formatted string platform independently
void log(const char* tag, const char* format, ...);

//! Terminates the application with a message. No leak checking.
[[noreturn]] void exitMsg(const char* tag,
                          const char* msg,
                          int         line,
                          const char* file);

//! Platform independent warn message output
void warnMsg(const char* tag,
             const char* msg,
             int         line,
             const char* file);

//! Platform independent error message output
void errorMsg(const char* tag,
              const char* msg,
              int         line,
              const char* file);

//! Shows the a spinner icon message
void showSpinnerMsg(string msg);

//! Hides the previous spinner icon message
void hideSpinnerMsg();

//! Returns in release config the max. NO. of threads otherwise 1
unsigned int maxThreads();

//////////////////////////////////
// Math Constants and Functions //
//////////////////////////////////

static const float PI        = 3.14159265358979f;
static const float RAD2DEG   = 180.0f / PI;
static const float DEG2RAD   = PI / 180.0f;
static const float TWOPI     = 2.0f * PI;
static const float ONEOVERPI = 1.0f / PI; // is faster than / PI
static const float HALFPI    = PI * 0.5f;

// clang-format off
template<class T> inline T sign(T a){return (T)((a > 0) ? 1 : (a < 0) ? -1 : 0);}
template<class T> inline T floor(T a){return (T)((int)a - ((a < 0 && a != (int)(a))));}
template<class T> inline T ceil(T a){return (T)((int)a + ((a > 0 && a != (int)(a))));}
template<class T> inline T fract(T a){return a - floor(a);}
template<class T> inline T abs(T a){return (a >= 0) ? a : -a;}
template<class T> inline T mod(T a, T b){return a - b * floor(a / b);}
template<class T> inline T step(T edge, T x){return (T)(x >= edge);}
template<class T> inline T pulse(T a, T b, T x){return (SL_step(a, x) - step(b, x));}
template<class T> inline T clamp(T a, T min, T max){return (a < min) ? min : (a > max) ? max : a;}
template<class T> inline T mix(T mix, T a, T b){return (1 - mix) * a + mix * b;}
template<class T> inline T lerp(T x, T a, T b){return (a + x * (b - a));}
//template<class T> inline T swap(T& a, T& b){T c = a; a = b; b = c;}
//-----------------------------------------------------------------------------
//! Returns true if a number is of power of 2
inline bool isPowerOf2(unsigned int a)
{
    return a == 1 || (a & (a - 1)) == 0;
}
//-----------------------------------------------------------------------------
//! Returns a uniform distributed random float number between min and max
inline float random(float min, float max)
{
    return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}
//-----------------------------------------------------------------------------
//! Returns a uniform distributed random int number between min and max
inline int random(int min, int max)
{
    return min + (rand() % (int)(max - min + 1));
}
//-----------------------------------------------------------------------------
//! Greatest common divisor of two integer numbers (ggT = grösster gemeinsame Teiler)
int gcd(int a, int b);
//-----------------------------------------------------------------------------
//! Returns the closest power of 2 to a passed number.
unsigned closestPowerOf2(unsigned num);
//-----------------------------------------------------------------------------
//! Returns the next power of 2 to a passed number.
unsigned nextPowerOf2(unsigned num);
//-----------------------------------------------------------------------------
// clang-format on

class ComputerInfos
{
public:
    static std::string user;
    static std::string name;
    static std::string brand;
    static std::string model;
    static std::string os;
    static std::string osVer;
    static std::string arch;
    static std::string id;

    static std::string get();
};

};
//-----------------------------------------------------------------------------

#endif
