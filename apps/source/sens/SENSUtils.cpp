#include "SENSUtils.h"
#include <Utils.h>

namespace SENS
{

void cropImage(cv::Mat& img, float targetWdivH, int& cropW, int& cropH)
{

    float inWdivH = (float)img.cols / (float)img.rows;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = targetWdivH < 0.0f ? inWdivH : targetWdivH;

    cropH = 0; // crop height in pixels of the source image
    cropW = 0; // crop width in pixels of the source image
    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)img.rows * outWdivH);
            height = img.rows;
            cropW  = (int)((float)(img.cols - width) * 0.5f);

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
            width  = img.cols;
            height = (int)((float)img.cols / outWdivH);
            cropH  = (int)((float)(img.rows - height) * 0.5f);

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

        img(cv::Rect(cropW, cropH, width, height)).copyTo(img);
        //imwrite("AfterCropping.bmp", lastFrame);
    }
}

void mirrorImage(cv::Mat& img, bool mirrorH, bool mirrorV)
{
    if (mirrorH)
    {
        cv::Mat mirrored;
        if (mirrorV)
            cv::flip(img, mirrored, -1);
        else
            cv::flip(img, mirrored, 1);
        img = mirrored;
    }
    else if (mirrorV)
    {
        cv::Mat mirrored;
        if (mirrorH)
            cv::flip(img, mirrored, -1);
        else
            cv::flip(img, mirrored, 0);
        img = mirrored;
    }
}

};
