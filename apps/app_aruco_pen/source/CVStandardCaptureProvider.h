//#############################################################################
//  File:      IDSPeakStandardCaptureProvider.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
#define SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H

#include <CVCaptureProvider.h>
#include <cv/CVTypedefs.h>
#include <cv/CVTypes.h>

//-----------------------------------------------------------------------------
//! The standard implementation of CVCaptureProvider
/*! CVStandardCaptureProvider is used to access data from OpenCV live captures.
 */
class CVStandardCaptureProvider : public CVCaptureProvider
{
private:
    SLint          _deviceIndex;   //!< the OpenCV device index
    CVVideoCapture _captureDevice; //!< the OpenCV video capture instance for accessing camera data

public:
    explicit CVStandardCaptureProvider(SLint deviceIndex, CVSize captureSize);
    ~CVStandardCaptureProvider() noexcept override;

    void   open() override;
    void   grab() override;
    void   close() override;
    SLbool isOpened() override;

    SLint deviceIndex() const { return _deviceIndex; }
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_CVSTANDARDCAPTUREPROVIDER_H
