//#############################################################################
//  File:      SLVRConvert.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVRCONVERT_H
#define SLPROJECT_SLVRCONVERT_H

#include <openvr.h>
#include <SLMat4.h>
#include <SLEnums.h>

//-----------------------------------------------------------------------------
//! SLVRConvert provides methods for converting between SL and OpenVR types
/*! The class has methods for converting OpenVR matrices to SL matrices and for
 * converting SL eyes to OpenVR eyes
 */
class SLVRConvert
{
public:
    static SLMat4f vrToSlMatrix(vr::HmdMatrix34_t matrix);
    static SLMat4f vrToSlMatrix(vr::HmdMatrix44_t matrix);

    static vr::Hmd_Eye slToVrEye(SLEyeType type);
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLVRCONVERT_H
