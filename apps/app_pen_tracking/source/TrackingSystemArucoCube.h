//#############################################################################
//  File:      TrackingSystemArucoCube.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_TRACKINGSYSTEMARUCOCUBE_H
#define SRC_TRACKINGSYSTEMARUCOCUBE_H

#include <TrackingSystem.h>
#include <CVMultiTracker.h>

//-----------------------------------------------------------------------------
class TrackingSystemArucoCube : public TrackingSystem
{
public:
    static constexpr float PEN_LENGTH = 0.147f - 0.025f + 0.002f;

    bool      track(CVCaptureProvider* provider) override;
    void      finalizeTracking() override;
    CVMatx44f worldMatrix() override;
    void      calibrate(CVCaptureProvider* provider) override;
    bool      isAcceptedProvider(CVCaptureProvider* provider) override;
    void      createPenNode() override;
    SLNode*   penNode() override { return _penNode; }
    float     penLength() { return PEN_LENGTH; }

    CVMultiTracker& multiTracker() { return _multiTracker; }

private:
    void optimizeTracking();

    CVMultiTracker _multiTracker;
    SLNode*        _penNode;
};
//-----------------------------------------------------------------------------
#endif // SRC_TRACKINGSYSTEMARUCOCUBE_H
