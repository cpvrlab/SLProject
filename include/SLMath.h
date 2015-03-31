//#############################################################################
//  File:      math/SLMath.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMATH_H
#define SLMATH_H

#include <SL.h>

//-----------------------------------------------------------------------------
// Math. constants
static const clock_t  CLOCKS_PER_HALFSEC = CLOCKS_PER_SEC>>1;

#ifdef SL_HAS_DOUBLE
static const SLdouble SL_PI = 3.1415926535897932384626433832795;
static const SLdouble SL_DOUBLE_MAX = DBL_MAX;    // max. double value
static const SLdouble SL_DOUBLE_MIN = DBL_MIN;    // min. double value
static const SLdouble SL_REAL_MAX = DBL_MAX;    // max. real value
static const SLdouble SL_REAL_MIN = DBL_MIN;    // min. real value
static const SLdouble SL_RAD2DEG  = 180.0f / SL_PI;
static const SLdouble SL_DEG2RAD  = SL_PI / 180.0f;
static const SLdouble SL_2PI      = 2.0f * SL_PI;
static const SLdouble SL_HALFPI   = SL_PI * 0.5f;
#else
static const SLfloat SL_PI = 3.14159265358979f;
static const SLfloat SL_REAL_MAX = FLT_MAX;    // max. real value
static const SLfloat SL_REAL_MIN = FLT_MIN;    // min. real value
static const SLfloat SL_RAD2DEG  = 180.0f / SL_PI;
static const SLfloat SL_DEG2RAD  = SL_PI / 180.0f;
static const SLfloat SL_2PI      = 2.0f * SL_PI;
static const SLfloat SL_HALFPI   = SL_PI * 0.5f;
#endif
//-----------------------------------------------------------------------------
// Misc. math functions
template<class T> inline T    SL_min   (T a,T b){return (a<b)? a : b;}
template<class T> inline T    SL_max   (T a,T b){return (a>b)? a : b;}
template<class T> inline T    SL_min   (T a,T b,T c){return (a<b && a<c)?a:(b<c)?b:c;}
template<class T> inline T    SL_max   (T a,T b,T c){return (a>b && a>c)?a:(b>c)?b:c;}
template<class T> inline T    SL_sign  (T a){return (T)((a>0) ? 1 : (a<0) ? -1 : 0);} 
template<class T> inline T    SL_floor (T a){return (T)((SLint)a - ((a<0 && a!=(SLint)(a))));}
template<class T> inline T    SL_ceil  (T a){return (T)((SLint)a + ((a>0 && a!=(SLint)(a))));}
template<class T> inline T    SL_fract (T a){return a-SL_floor(a);} 
template<class T> inline T    SL_abs   (T a){return (a>=0) ? a : -a;}
template<class T> inline T    SL_mod   (T a, T b){return a - b*SL_floor(a/b);} 
template<class T> inline T    SL_step  (T edge,T x){return (T) (x>=edge);}
template<class T> inline T    SL_pulse (T a,T b,T x){return (SL_step(a,x)-SL_step(b,x));} 
template<class T> inline T    SL_clamp (T a,T min,T max){return (a<min) ? min : (a>max) ? max : a;} 
template<class T> inline T    SL_mix   (T mix,T a,T b){return (1-mix)*a + mix*b;}
template<class T> inline T    SL_lerp  (T x,T a,T b){return (a + x*(b-a));} 
template<class T> inline void SL_swap  (T& a,T& b){T c=a; a=b; b=c;}
//-----------------------------------------------------------------------------
inline SLbool SL_isPowerOf2(SLuint a)
{
    return a == 1 || (a & (a-1)) == 0;
}
//-----------------------------------------------------------------------------
inline SLfloat SL_random(SLfloat min, SLfloat max)
{
    return ((SLfloat)rand()/(SLfloat)RAND_MAX) * (max-min) + min;
}
//-----------------------------------------------------------------------------
#endif
