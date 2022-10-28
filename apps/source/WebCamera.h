// #############################################################################
//   File:      WebCamera.h
//   Purpose:   Interface to access the camera through the browser.
//   Date:      October 2022
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marino von Wattenwyl
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#ifndef SLPROJECT_WEBCAMERA_H
#define SLPROJECT_WEBCAMERA_H

#include <CVTypedefs.h>

//-----------------------------------------------------------------------------
class WebCamera
{
public:
    void  open();
    CVMat read();
    void  close();

    bool isOpened() { return _isOpened; }

private:
    bool  _isOpened = false;
    CVMat _image;
    bool  _isImageAllocated = false;
};
//-----------------------------------------------------------------------------
#endif // SLPROJECT_WEBCAMERA_H
