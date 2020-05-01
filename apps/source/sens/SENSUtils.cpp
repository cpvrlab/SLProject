#include "SENSUtils.h"
#include <Utils.h>
#include <HighResTimer.h>

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

//opposite to crop image: extrend
void extendWithBars(cv::Mat& img, float targetWdivH, int cvBorderType, int& addW, int& addH)
{
    //HighResTimer t;
    float inWdivH  = (float)img.cols / (float)img.rows;
    float outWdivH = targetWdivH < 0.0f ? inWdivH : targetWdivH;

    addH = 0;
    addW = 0;
    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int width  = 0; // width in pixels of the destination image
        int height = 0; // height in pixels of the destination image
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // add bar bottom and top (old: crop input image left & right)
        {
            width  = img.cols;
            height = (int)((float)img.cols / outWdivH);
            addH   = (int)((float)(height - img.rows) * 0.5f);

            // Height must be devidable by 4
            hModulo4 = height % 4;
            if (hModulo4 == 1) height--;
            if (hModulo4 == 2)
            {
                addH++;
                height -= 2;
            }
            if (hModulo4 == 3) height++;
        }
        else // add bar left and right (old: crop input image at top & bottom)
        {
            width  = (int)((float)img.rows * outWdivH);
            height = img.rows;
            addW   = (int)((float)(width - img.cols) * 0.5f);

            // Width must be devidable by 4
            wModulo4 = width % 4;
            if (wModulo4 == 1) width--;
            if (wModulo4 == 2)
            {
                addW++;
                width -= 2;
            }
            if (wModulo4 == 3) width++;
        }

        int        borderType = cv::BORDER_CONSTANT;
        cv::Scalar value(0, 0, 0);
        copyMakeBorder(img, img, addH, addH, addW, addW, cvBorderType, value);
        //cv::imwrite("AfterExtendWithBars.bmp", img);

        //if (cvBorderType == cv::BORDER_REPLICATE)
        //{
        //    if (addH > 0)
        //    {
        //    }
        //    else if (addW > 0)
        //    {

        //        cv::Size iS = img.size();
        //        //left
        //        cv::Rect barRoiL(0, 0, addW, iS.height);
        //        cv::Mat  barImgL = img(barRoiL);
        //        //cv::imwrite("barleftBefore.bmp", barImgL);
        //        cv::blur(barImgL, barImgL, cv::Size(1, 11), cv::Point(-1, -1));
        //        //cv::imwrite("barleftAfter.bmp", barImgL);
        //        //right
        //        cv::Rect barRoiR(iS.width - addW, 0, addW, iS.height);
        //        cv::Mat  barImgR = img(barRoiR);
        //        //cv::imwrite("barRightBefore.bmp", barImgR);
        //        cv::blur(barImgR, barImgR, cv::Size(1, 11), cv::Point(-1, -1));
        //        //cv::imwrite("barRightAfter.bmp", barImgR);
        //    }
        //}
    }
    //Utils::log("extendWithBars", "elapsed time %f ms", t.elapsedTimeInMilliSec());
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
