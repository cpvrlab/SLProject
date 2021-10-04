//#############################################################################
//  File:      CVCaptureProvider.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "CVCaptureProvider.h"

#include <utility>

//-----------------------------------------------------------------------------
CVCaptureProvider::CVCaptureProvider(SLstring uid,
                                     SLstring name,
                                     CVSize   captureSize)
  : _uid(std::move(uid)),
    _name(std::move(name)),
    _camera(CVCameraType::FRONTFACING),
    _captureSize(std::move(captureSize))
{
}
//-----------------------------------------------------------------------------