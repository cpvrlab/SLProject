//#############################################################################
//  File:      CVCaptureProviderSpryTrack.h
//  Date:      December 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SRC_CVCAPTUREPROVIDERSPRYTRACK_H
#define SRC_CVCAPTUREPROVIDERSPRYTRACK_H

#include <CVCaptureProvider.h>
#include <SpryTrackDevice.h>

//-----------------------------------------------------------------------------
class CVCaptureProviderSpryTrack : public CVCaptureProvider
{
private:
    SpryTrackDevice _device;
    SLbool          _isOpened = false;

public:
    explicit CVCaptureProviderSpryTrack(CVSize captureSize);
    ~CVCaptureProviderSpryTrack() noexcept override;

    SpryTrackDevice& device() { return _device; }

    void   open() override;
    void   grab() override;
    void   close() override;
    SLbool isOpened() override;
};
//-----------------------------------------------------------------------------
#endif // SRC_CVCAPTUREPROVIDERSPRYTRACK_H
