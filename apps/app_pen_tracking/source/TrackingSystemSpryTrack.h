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
#include <CVCaptureProviderSpryTrack.h>

//-----------------------------------------------------------------------------
class TrackingSystemSpryTrack : public TrackingSystem
{
public:
    static constexpr float PEN_LENGTH = 0.13f + 0.015f - 0.0015f;

    bool      track(CVCaptureProvider* provider) override;
    void      finalizeTracking() override;
    CVMatx44f worldMatrix() override;
    void      calibrate(CVCaptureProvider* provider) override;
    bool      isAcceptedProvider(CVCaptureProvider* provider) override;
    void      createPenNode() override;
    SLNode*   penNode() override { return _penNode; }
    float     penLength() override { return PEN_LENGTH; }

    CVMatx44f extrinsicMat() { return _extrinsicMat; }
    CVMatx44f markerMat() { return _markerMat; }

private:
    static SpryTrackDevice& getDevice(CVCaptureProvider* provider);

private:
    CVMatx44f _worldMatrix;
    CVMatx44f _extrinsicMat;
    CVMatx44f _markerMat;
    SLNode*   _penNode;
};
//-----------------------------------------------------------------------------
#endif // SRC_TRACKINGSYSTEMSPRYTRACK_H
