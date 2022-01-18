//#############################################################################
//  File:      CoordSystemConversions.h
//  Date:      January 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_COORDSYSTEMCONVERSIONS_H
#define SRC_COORDSYSTEMCONVERSIONS_H

#include <cstring>

//-----------------------------------------------------------------------------
namespace CoordSystemConversions
{

enum Axis
{
    POS_X,
    POS_Y,
    POS_Z,
    NEG_X,
    NEG_Y,
    NEG_Z
};
//-----------------------------------------------------------------------------
template<Axis Axis,
         typename T>
float getComponentFromAxis(T sx, T sy, T sz)
{
    // clang-format off
    switch (Axis)
    {
        case POS_X: return  sx;
        case POS_Y: return  sy;
        case POS_Z: return  sz;
        case NEG_X: return -sx;
        case NEG_Y: return -sy;
        case NEG_Z: return -sz;
        default: return 0.0f;
    }
    // clang-format on
}
//-----------------------------------------------------------------------------
template<Axis DestXInSrc,
         Axis DestYInSrc,
         Axis DestZInSrc,
         typename TSrc,
         typename TDest>
void swapAxes(TSrc sx, TSrc sy, TSrc sz, TDest* dx, TDest* dy, TDest* dz)
{
    *dx = (TDest)getComponentFromAxis<DestXInSrc>(sx, sy, sz);
    *dy = (TDest)getComponentFromAxis<DestYInSrc>(sx, sy, sz);
    *dz = (TDest)getComponentFromAxis<DestZInSrc>(sx, sy, sz);
}
//-----------------------------------------------------------------------------
template<Axis DestXInSrc,
         Axis DestYInSrc,
         Axis DestZInSrc,
         bool IsOldColumnMajor,
         bool IsNewColumnMajor,
         typename TSrc,
         typename TDest>
void convert4x4(const TSrc* src, TDest* dest)
{
    // The source data in column-major order
    TSrc temp[16];

    // Copy all data from src to temp and transpose if it is not column-major
    if (!IsOldColumnMajor)
    {
        // clang-format off
        temp[ 0] = src[ 0]; temp[ 4] = src[ 1]; temp[ 8] = src[ 2]; temp[12] = src[ 3];
        temp[ 1] = src[ 4]; temp[ 5] = src[ 5]; temp[ 9] = src[ 6]; temp[13] = src[ 7];
        temp[ 2] = src[ 8]; temp[ 6] = src[ 9]; temp[10] = src[10]; temp[14] = src[11];
        temp[ 3] = src[12]; temp[ 7] = src[13]; temp[11] = src[14]; temp[15] = src[15];
        // clang-format on
    }
    else
    {
        std::memcpy(temp, src, 16 * sizeof(TSrc));
    }

    // Copy all data from temp to src and modify axes
    if (IsNewColumnMajor)
    {
        // clang-format off
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 0], temp[ 1], temp[ 2], dest +  0, dest +  1, dest +  2);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 4], temp[ 5], temp[ 6], dest +  4, dest +  5, dest +  6);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 8], temp[ 9], temp[10], dest +  8, dest +  9, dest + 10);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[12], temp[13], temp[14], dest + 12, dest + 13, dest + 14);
        dest[3] = dest[7] = dest[11] = 0.0f;
        dest[15]                     = 1.0f;
        // clang-format on
    }
    else
    {
        // clang-format off
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 0], temp[ 1], temp[ 2], dest +  0, dest +  4, dest +  8);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 4], temp[ 5], temp[ 6], dest +  1, dest +  5, dest +  9);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[ 8], temp[ 9], temp[10], dest +  2, dest +  6, dest + 10);
        swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(temp[12], temp[13], temp[14], dest +  3, dest +  7, dest + 11);
        dest[12] = dest[13] = dest[14] = 0.0f;
        dest[15]                       = 1.0f;
        // clang-format on
    }
}
//-----------------------------------------------------------------------------
template<Axis DestXInSrc,
         Axis DestYInSrc,
         Axis DestZInSrc,
         typename TSrc,
         typename TDest>
void convert3(const TSrc* src, TDest* dest)
{
    swapAxes<DestXInSrc, DestYInSrc, DestZInSrc>(src[0], src[1], src[2], dest + 0, dest + 1, dest + 2);
}
//-----------------------------------------------------------------------------
void cv2gl4x4f(float* src, float* dest);
//-----------------------------------------------------------------------------

} // namespace CoordSystemConversions
//-----------------------------------------------------------------------------
#endif // SRC_COORDSYSTEMCONVERSIONS_H
