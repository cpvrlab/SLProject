//#############################################################################
//  File:      math/SLAlgo.cpp
//  Purpose:   Container for general algorithm functions
//  Date:      November
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Michael Goettlicher, Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLAlgo.h>
#include <cassert>

namespace SLAlgo
{

bool estimateHorizon(const SLMat3f& enuRs, const SLMat3f& sRc, SLVec3f& horizon)
{
    SLMat3f cRenu = (enuRs * sRc).transposed();
    //estimate horizon in camera frame:
    //-normal vector of camera x-y-plane in enu frame definition: this is the camera z-axis epressed in enu frame
    SLVec3f normalCamXYPlane = SLVec3f(0, 0, 1);
    //-normal vector of enu x-y-plane in camera frame: this is the enu z-axis rotated into camera coord. frame
    SLVec3f normalEnuXYPlane = cRenu * SLVec3f(0, 0, 1);
    //-Estimation of intersetion line (horizon):
    //Then the crossproduct of both vectors defines the direction of the intersection line. In our special case we know that the origin is a point that lies on both planes.
    //Then origin together with the direction vector define the horizon.
    horizon.cross(normalEnuXYPlane, normalCamXYPlane);

    //check that vectors are not parallel
    float l = horizon.length();
    if (l < 0.01f)
    {
        horizon = {1.f, 0.f, 0.f};
        return false;
    }
    else
    {
        horizon /= l;
        return true;
    }
}

template<typename T>
T geoDegMinSec2Decimal(int degrees, int minutes, T seconds)
{
    return (T)degrees + ((T)(minutes * 60) + seconds) / ((T)3600);
}
//explicit template instantiation for float and double (only these make sense)
template float  geoDegMinSec2Decimal(int degrees, int minutes, float seconds);
template double geoDegMinSec2Decimal(int degrees, int minutes, double seconds);

// clang-format off
template<typename T>
SLVec3<T> geoDegMinSec2Decimal(int degreesLat, int minutesLat, T secondsLat,
                               int degreesLon, int minutesLon, T secondsLon,
                               T altM)
{
    //https://www.koordinaten-umrechner.de/
    assert(degreesLat > -90 && degreesLat < 90);
    assert(degreesLon > -180 && degreesLon < 180);
    assert(minutesLat > 0 && minutesLat < 60);
    assert(minutesLon > 0 && minutesLon < 60);
    assert(secondsLat >= (T)0 && secondsLat < (T)60);
    assert(secondsLon >= (T)0 && secondsLon < (T)60);
    
    SLVec3<T> vec;
    vec.x = geoDegMinSec2Decimal<T>(degreesLat, minutesLat, secondsLat);
    vec.y = geoDegMinSec2Decimal<T>(degreesLon, minutesLon, secondsLon);
    vec.z = altM;
    
    return vec;
}
//explicit template instantiation for float and double (only these make sense)
template SLVec3f geoDegMinSec2Decimal(int degreesLat, int minutesLat, float secondsLat,
                                      int degreesLon, int minutesLon, float secondsLon,
                                      float altM);
template SLVec3d geoDegMinSec2Decimal(int degreesLat, int minutesLat, double secondsLat,
                                      int degreesLon, int minutesLon, double secondsLon,
                                      double altM);
// clang-format on
};
