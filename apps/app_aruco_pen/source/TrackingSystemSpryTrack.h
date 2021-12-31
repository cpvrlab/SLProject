//#############################################################################
//  File:      TrackingSystemSpryTrack.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_TRACKINGSYSTEMSPRYTRACK_H
#define SRC_TRACKINGSYSTEMSPRYTRACK_H

#include <TrackingSystem.h>

//-----------------------------------------------------------------------------
class TrackingSystemSpryTrack : public TrackingSystem
{
public:
    bool      track(CVCaptureProvider* provider) override;
    void      finalizeTracking() override;
    CVMatx44f worldMatrix() override;

private:
    CVMatx44f _worldMatrix;
};
//-----------------------------------------------------------------------------
#endif // SRC_TRACKINGSYSTEMSPRYTRACK_H
