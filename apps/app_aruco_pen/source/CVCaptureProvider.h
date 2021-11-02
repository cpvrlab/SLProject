//#############################################################################
//  File:      CVCaptureProvider.h
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_CVCAPTUREPROVIDER_H
#define SLPROJECT_CVCAPTUREPROVIDER_H

#include <cv/CVTypedefs.h>
#include <cv/CVCamera.h>
#include <SL.h>

#include <utility>

//-----------------------------------------------------------------------------
//! Interface for accessing capture data from cameras, files, etc.
/*! CVCaptureProvider generalizes access to video capture by exposing
 * functions that all capture sources have in common.
 * These functions could then for example be called
 * to display the capture on screen or to perform tracking.
 * The advantage of an independent interface is that the capture source
 * can easily be replaced by instantiating a different implementation
 * of CVCaptureProvider.
 */
class CVCaptureProvider
{
private:
    SLstring _uid;    //!< the unique identifier for this capture provider
    SLstring _name;   //!< human-readable name intended for displaying
    CVCamera _camera; //!< camera object for tracking, mirroring, distortion, etc.

protected:
    CVMat  _lastFrameBGR;  //!< the last grabbed frame in the BGR format
    CVMat  _lastFrameGray; //!< the gray version of the last grabbed frame
    CVSize _captureSize;   //!< width and height of the capture in pixels

public:
    CVCaptureProvider(SLstring uid,
                      SLstring name,
                      CVSize   captureSize);

    virtual ~CVCaptureProvider() noexcept = default;
    virtual void   open()                 = 0;
    virtual void   grab()                 = 0;
    virtual void   close()                = 0;
    virtual SLbool isOpened()             = 0;

    // Getters
    SLstring  uid() { return _uid; }
    SLstring  name() { return _name; }
    CVCamera& camera() { return _camera; }
    CVMat&    lastFrameBGR() { return _lastFrameBGR; }
    CVMat&    lastFrameGray() { return _lastFrameGray; }
    CVSize    captureSize() { return _captureSize; }
    CVSize    lastFrameSize() { return {_lastFrameBGR.cols, _lastFrameBGR.cols}; }

    void cropToAspectRatio(float aspectRatio);

private:
    static void cropToAspectRatio(CVMat& image, float aspectRatio);
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_CVCAPTUREPROVIDER_H
