//#############################################################################
//  File:      SLVRConvert.cpp
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <vr/SLVRConvert.h>
#include <vr/SLVR.h>

//-----------------------------------------------------------------------------
/*! Converts an OpenVR 3x4 matrix to a SLProject matrix
 * @param matrix The OpenVR matrix
 * @return The converted SLProject matrix
 */
SLMat4f SLVRConvert::openVRMatrixToSLMatrix(vr::HmdMatrix34_t matrix)
{
    SLMat4f result;

    // First column
    result.m(0, matrix.m[0][0]);
    result.m(1, matrix.m[1][0]);
    result.m(2, matrix.m[2][0]);
    result.m(3, 0);

    // Second column
    result.m(4, matrix.m[0][1]);
    result.m(5, matrix.m[1][1]);
    result.m(6, matrix.m[2][1]);
    result.m(7, 0);

    // Third column
    result.m(8, matrix.m[0][2]);
    result.m(9, matrix.m[1][2]);
    result.m(10, matrix.m[2][2]);
    result.m(11, 0);

    // Fourth column
    result.m(12, matrix.m[0][3]);
    result.m(13, matrix.m[1][3]);
    result.m(14, matrix.m[2][3]);
    result.m(15, 1);

    return result;
}
//-----------------------------------------------------------------------------
/*! Converts an OpenVR 4x4 matrix to a SLProject matrix
 * @param matrix The OpenVR matrix
 * @return The converted SLProject matrix
 */
SLMat4f SLVRConvert::openVRMatrixToSLMatrix(vr::HmdMatrix44_t matrix)
{
    SLMat4f result;

    // First column
    result.m(0, matrix.m[0][0]);
    result.m(1, matrix.m[1][0]);
    result.m(2, matrix.m[2][0]);
    result.m(3, matrix.m[3][0]);

    // Second column
    result.m(4, matrix.m[0][1]);
    result.m(5, matrix.m[1][1]);
    result.m(6, matrix.m[2][1]);
    result.m(7, matrix.m[3][1]);

    // Third column
    result.m(8, matrix.m[0][2]);
    result.m(9, matrix.m[1][2]);
    result.m(10, matrix.m[2][2]);
    result.m(11, matrix.m[3][2]);

    // Fourth column
    result.m(12, matrix.m[0][3]);
    result.m(13, matrix.m[1][3]);
    result.m(14, matrix.m[2][3]);
    result.m(15, matrix.m[3][3]);

    return result;
}
//-----------------------------------------------------------------------------
/*! Converts a SLEyeType value to a OpenVR eye value
 * @param type The SLEyeType value
 * @return The converted OpenVR eye value
 */
vr::Hmd_Eye SLVRConvert::SLEyeTypeToOpenVREye(SLEyeType type)
{
    if (type == ET_left)
        return vr::Hmd_Eye::Eye_Left;
    else if (type == ET_right)
        return vr::Hmd_Eye::Eye_Right;
    else
    {
        VR_WARNING("Invalid eye type specified")
        return vr::Hmd_Eye::Eye_Left;
    }
}
//-----------------------------------------------------------------------------