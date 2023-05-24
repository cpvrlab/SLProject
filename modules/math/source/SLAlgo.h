//#############################################################################
//  File:      math/SLAlgo.h
//  Purpose:   Container for general algorithm functions
//  Date:      November
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLALGO_H
#define SLALGO_H

#include <SLMath.h>
#include <SLMat3.h>
#include <SLVec3.h>

//-----------------------------------------------------------------------------
namespace SLAlgo
{
bool estimateHorizon(const SLMat3f& enuRs, const SLMat3f& sRc, SLVec3f& horizon);

//! convert geodetic datum defined in degrees, minutes and seconds to decimal
template<typename T>
T geoDegMinSec2Decimal(int degrees, int minutes, T seconds);

//! Latitude Longitude Altitude (LatLonAlt), defined in Degrees, Minutes, Secondes format to decimal
template<typename T>
SLVec3<T> geoDegMinSec2Decimal(int degreesLat, int minutesLat, T secondsLat, int degreesLon, int minutesLon, T secondsLon, T alt);

};
//-----------------------------------------------------------------------------
#endif
