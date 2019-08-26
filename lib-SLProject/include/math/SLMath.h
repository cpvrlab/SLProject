//#############################################################################
//  File:      math/SLMath.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLMATH_H
#define SLMATH_H

#include <SL.h>
#include <cfloat>

//-----------------------------------------------------------------------------
// Math. constants
static const clock_t CLOCKS_PER_HALFSEC = CLOCKS_PER_SEC >> 1;

#ifdef SL_HAS_DOUBLE
static const SLdouble SL_PI         = 3.1415926535897932384626433832795;
static const SLdouble SL_DOUBLE_MAX = DBL_MAX; // max. double value
static const SLdouble SL_DOUBLE_MIN = DBL_MIN; // min. double value
static const SLdouble SL_REAL_MAX   = DBL_MAX; // max. real value
static const SLdouble SL_REAL_MIN   = DBL_MIN; // min. real value
static const SLdouble SL_RAD2DEG    = 180.0f / SL_PI;
static const SLdouble SL_DEG2RAD    = SL_PI / 180.0f;
static const SLdouble SL_2PI        = 2.0f * SL_PI;
static const SLdouble SL_HALFPI     = SL_PI * 0.5f;
#else
static const SLfloat SL_PI       = 3.14159265358979f;
static const SLfloat SL_REAL_MAX = FLT_MAX; // max. real value
static const SLfloat SL_REAL_MIN = FLT_MIN; // min. real value
static const SLfloat SL_RAD2DEG  = 180.0f / SL_PI;
static const SLfloat SL_DEG2RAD  = SL_PI / 180.0f;
static const SLfloat SL_2PI      = 2.0f * SL_PI;
static const SLfloat SL_HALFPI   = SL_PI * 0.5f;
#endif

// Some constants for ecef to lla converstions
static const double SL_EARTH_RADIUS_A          = 6378137;
static const double SL_EARTH_ECCENTRICTIY      = 8.1819190842622e-2;
static const double SL_EARTH_RADIUS_A_SQR      = SL_EARTH_RADIUS_A * SL_EARTH_RADIUS_A;
static const double SL_EARTH_ECCENTRICTIY_SQR  = SL_EARTH_ECCENTRICTIY * SL_EARTH_ECCENTRICTIY;
static const double SL_EARTH_RADIUS_B          = sqrt(SL_EARTH_RADIUS_A_SQR * (1 - SL_EARTH_ECCENTRICTIY_SQR));
static const double SL_EARTH_RADIUS_B_SQR      = SL_EARTH_RADIUS_B * SL_EARTH_RADIUS_B;
static const double SL_EARTH_ECCENTRICTIY2     = sqrt((SL_EARTH_RADIUS_A_SQR - SL_EARTH_RADIUS_B_SQR) / SL_EARTH_RADIUS_B_SQR);
static const double SL_EARTH_ECCENTRICTIY2_SQR = SL_EARTH_ECCENTRICTIY2 * SL_EARTH_ECCENTRICTIY2;

//-----------------------------------------------------------------------------
#endif
