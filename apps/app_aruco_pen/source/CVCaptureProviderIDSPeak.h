//#############################################################################
//  File:      CVCaptureProviderIDSPeak.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVCAPTUREPROVIDERIDSPEAK_H
#define SLPROJECT_CVCAPTUREPROVIDERIDSPEAK_H

#include <CVCaptureProvider.h>

//-----------------------------------------------------------------------------
//! Implementation of CVCaptureProvider for IDS Peak cameras
/*! This implementation of CVCaptureProvider uses the IDS Peak library
 * to access camera data.
 */
class CVCaptureProviderIDSPeak : public CVCaptureProvider
{
private:
    SLint  _deviceIndex;      //!< the index of this device in the IDS peak device list
    SLbool _isOpened = false; //!< tracks whether or not the device is opened

public:
    explicit CVCaptureProviderIDSPeak(SLint deviceIndex, CVSize captureSize);
    ~CVCaptureProviderIDSPeak() noexcept override;

    void   open() override;
    void   grab() override;
    void   close() override;
    SLbool isOpened() override;

    SLint deviceIndex() const { return _deviceIndex; }
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_CVCAPTUREPROVIDERIDSPEAK_H