//#############################################################################
//  File:      math/SLMath.h
//  Purpose:   Container for general algorithm functions
//  Date:      November 2020
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Micheal Goettlicher
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMATH_H
#define SLMATH_H

#include <vector>
#include <string>

//-----------------------------------------------------------------------------
// Redefinition of standard types for platform independence
typedef std::string SLstring;
#ifndef SL_OS_ANDROID
typedef std::wstring SLwstring;
#endif
typedef char           SLchar;     // analog to GLchar (char is signed [-128 ... 127]!)
typedef unsigned char  SLuchar;    // analog to GLuchar
typedef signed long    SLlong;     // analog to GLlong
typedef unsigned long  SLulong;    // analog to GLulong
typedef signed char    SLbyte;     // analog to GLbyte
typedef unsigned char  SLubyte;    // analog to GLubyte
typedef short          SLshort;    // analog to GLshort
typedef unsigned short SLushort;   // analog to GLushort
typedef int            SLint;      // analog to GLint
typedef unsigned int   SLuint;     // analog to GLuint
typedef int            SLsizei;    // analog to GLsizei
typedef float          SLfloat;    // analog to GLfloat
typedef double         SLdouble;   // analog to GLdouble
typedef bool           SLbool;     // analog to GLbool
typedef unsigned int   SLenum;     // analog to GLenum
typedef unsigned int   SLbitfield; // analog to GLbitfield
typedef int64_t        SLint64;
typedef uint64_t       SLuint64;

// All 1D vectors begin with SLV*
typedef std::vector<SLbool>   SLVbool;
typedef std::vector<SLbyte>   SLVbyte;
typedef std::vector<SLubyte>  SLVubyte;
typedef std::vector<SLchar>   SLVchar;
typedef std::vector<SLuchar>  SLVuchar;
typedef std::vector<SLshort>  SLVshort;
typedef std::vector<SLushort> SLVushort;
typedef std::vector<SLint>    SLVint;
typedef std::vector<SLuint>   SLVuint;
typedef std::vector<SLlong>   SLVlong;
typedef std::vector<SLulong>  SLVulong;
typedef std::vector<SLfloat>  SLVfloat;
typedef std::vector<SLstring> SLVstring;
typedef std::vector<size_t>   SLVsize_t;

// All 2D vectors begin with SLVV*
typedef std::vector<std::vector<SLfloat>>  SLVVfloat;
typedef std::vector<std::vector<SLuchar>>  SLVVuchar;
typedef std::vector<std::vector<SLushort>> SLVVushort;
typedef std::vector<std::vector<SLuint>>   SLVVuint;
typedef std::vector<std::vector<SLchar>>   SLVVchar;
typedef std::vector<std::vector<SLshort>>  SLVVshort;
typedef std::vector<std::vector<SLint>>    SLVVint;

//-----------------------------------------------------------------------------
// Some debugging and error handling macros
#define SLMATH_LOG(...) Utils::log("SLProject", __VA_ARGS__)
#define SLMATH_EXIT_MSG(message) Utils::exitMsg("SLProject", (message), __LINE__, __FILE__)
#define SLMATH_WARN_MSG(message) Utils::warnMsg("SLProject", (message), __LINE__, __FILE__)

#endif
