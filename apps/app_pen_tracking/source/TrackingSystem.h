//#############################################################################
//  File:      TrackingSystem.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_TRACKINGSYSTEM_H
#define SRC_TRACKINGSYSTEM_H

#include <CVTypedefs.h>
#include <CVCaptureProvider.h>

//-----------------------------------------------------------------------------
class TrackingSystem
{
public:
    virtual ~TrackingSystem() = default;
    virtual bool      track(CVCaptureProvider* provider)              = 0;
    virtual void      finalizeTracking()                              = 0;
    virtual CVMatx44f worldMatrix()                                   = 0;
    virtual void      calibrate(CVCaptureProvider* provider)          = 0;
    virtual bool      isAcceptedProvider(CVCaptureProvider* provider) = 0;
};
//-----------------------------------------------------------------------------
#endif // SRC_TRACKINGSYSTEM_H
