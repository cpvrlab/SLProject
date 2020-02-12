//#############################################################################
//  File:      Utils.h
//  Author:    Marcus Hudritsch
//  Date:      May 2019
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CPLVRLAB_UTILS_H
#define CPLVRLAB_UTILS_H

#include <string>
#include <vector>
#include <cfloat>
#include <memory>
#include <FileLog.h>

using namespace std;

//class FileLog;
//-----------------------------------------------------------------------------
//! Utils provides utility functions
/*!
 Function are grouped into sections:
 - String Handling Functions
 - File Handling Functions
 - Logging Functions
 - Network Handling Functions
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

//! Trims a string at the end
string trimString(const string& s, const string& drop = " ");

//! Splits an input string at a delimeter character into a string vector
void splitString(const string& s, char delimiter, vector<string>& splits);

//! Replaces in the source string the from string by the to string
void replaceString(string& source, const string& from, const string& to);

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

//! Returns the inputDir string with unified forward slashes, e.g.: "dirA/dirB/"
string unifySlashes(const string& inputDir);

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

//! Returns the file extension without dot in lower case
string getFileExt(const string& filename);

//! Returns a vector directory names with path in dir
vector<string> getDirNamesInDir(const string& dirName);

//! Returns a vector of sorted names (files and directories) with path in dir
vector<string> getAllNamesInDir(const string& dirName);

//! Returns a vector of storted filesnames in dirName
vector<string> getFileNamesInDir(const string& dirName);

//! Returns true if a directory exists.
bool dirExists(const string& path);

//! Returns the file size in bytes
unsigned int getFileSize(const string& filename);

//! Creates a directory with given path
bool makeDir(const string& path);

//! RemoveDir deletes a directory with given path
void removeDir(const string& path);

//! Returns true if a file exists.
bool fileExists(const string& pathfilename);

//! Returns the writable configuration directory
string getAppsWritableDir();

//! Returns the working directory
string getCurrentWorkingDir();

//! Deletes a file on the filesystem
bool deleteFile(string& pathfilename);

///////////////////////
// Logging Functions //
///////////////////////
//! FileLog Instance for logging to logfile. If it is instantiated the logging methods
//! will also output into this file. Instantiate it with initFileLog function.
static std::unique_ptr<FileLog> fileLog;
//! Instantiates FileLog instance
void initFileLog(const std::string logDir, bool forceFlush);

//! logs a formatted string platform independently
void log(const char* tag, const char* format, ...);

//! Terminates the application with a message. No leak cheching.
[[noreturn]] void exitMsg(const char* tag,
                          const char* msg,
                          int         line,
                          const char* file);

void warnMsg(const char* tag,
             const char* msg,
             int         line,
             const char* file);

void errorMsg(const char* tag,
              const char* msg,
              int         line,
              const char* file);

//! Returns in release config the max. NO. of threads otherwise 1
unsigned int maxThreads();

////////////////////////////////
// Network Handling Functions //
////////////////////////////////

//! Download a file from an http url into the outFile
uint64_t httpGet(const string& httpURL, const string& outFolder = "");

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
inline bool isPowerOf2(unsigned int a)
{
    return a == 1 || (a & (a - 1)) == 0;
}
//-----------------------------------------------------------------------------
inline float random(float min, float max)
{
    return ((float)rand() / (float)RAND_MAX) * (max - min) + min;
}
//-----------------------------------------------------------------------------
inline int pow(int x, int p)
{
    if (p == 0) return 1;
    if (p == 1) return x;
    return x * pow(x, p - 1);
}
//-----------------------------------------------------------------------------
//! Greatest common divisor of two integer numbers (ggT = grösster gemeinsame Teiler)
int gcd(int a, int b);
//-----------------------------------------------------------------------------
//! Lowest common multiple (kgV = kleinstes gemeinsames Vielfache)
int lcm(int a, int b);
//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// clang-format on
#endif
