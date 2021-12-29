//#############################################################################
//  File:      CVCaptureProvider.cpp
//  Date:      October 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVCaptureProvider.h>
#include <Instrumentor.h>

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
void CVCaptureProvider::cropToAspectRatio(float aspectRatio)
{
    PROFILE_FUNCTION();

    cropToAspectRatio(_lastFrameBGR, aspectRatio);
    cropToAspectRatio(_lastFrameGray, aspectRatio);
}
//-----------------------------------------------------------------------------
void CVCaptureProvider::cropToAspectRatio(CVMat& image, float aspectRatio)
{
    PROFILE_FUNCTION();

    float inWdivH = (float)image.cols / (float)image.rows;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = aspectRatio < 0.0f ? inWdivH : aspectRatio;

    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int cropH  = 0; // crop height in pixels of the source image
        int cropW  = 0; // crop width in pixels of the source image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)image.rows * outWdivH);
            height = image.rows;
            cropW  = (int)((float)(image.cols - width) * 0.5f);

            // Width must be devidable by 4
            wModulo4 = width % 4;
            if (wModulo4 == 1) width--;
            if (wModulo4 == 2)
            {
                cropW++;
                width -= 2;
            }
            if (wModulo4 == 3) width++;
        }
        else // crop input image at top & bottom
        {
            width  = image.cols;
            height = (int)((float)image.cols / outWdivH);
            cropH  = (int)((float)(image.rows - height) * 0.5f);

            // Height must be devidable by 4
            hModulo4 = height % 4;
            if (hModulo4 == 1) height--;
            if (hModulo4 == 2)
            {
                cropH++;
                height -= 2;
            }
            if (hModulo4 == 3) height++;
        }

        image(CVRect(cropW, cropH, width, height)).copyTo(image);
    }
}
//-----------------------------------------------------------------------------
