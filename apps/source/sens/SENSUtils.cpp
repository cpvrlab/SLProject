#include "SENSUtils.h"
#include <Utils.h>

namespace SENS
{

bool calcCrop(cv::Size inputSize, float targetWdivH, int& cropW, int& cropH, int& width, int&height)
{
    cropH = 0; // crop height in pixels of the source image
    cropW = 0; // crop width in pixels of the source image
    width  = inputSize.width; // width in pixels of the destination image
    height = inputSize.height; // height in pixels of the destination image
    
    float inWdivH = (float)inputSize.width / (float)inputSize.height;
    // viewportWdivH is negative the viewport aspect will be the same
    float outWdivH = targetWdivH < 0.0f ? inWdivH : targetWdivH;
    if (Utils::abs(inWdivH - outWdivH) > 0.01f)
    {
        int wModulo4;
        int hModulo4;

        if (inWdivH > outWdivH) // crop input image left & right
        {
            width  = (int)((float)inputSize.height * outWdivH);
            height = inputSize.height;
            cropW  = (int)((float)(inputSize.width - width) * 0.5f);

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
            width  = inputSize.width;
            height = (int)((float)inputSize.width / outWdivH);
            cropH  = (int)((float)(inputSize.height - height) * 0.5f);

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
        return true;
    }
    else
        return false;
}

void cropImage(cv::Mat& img, float targetWdivH, int& cropW, int& cropH)
{
    int width, height;
    if(calcCrop(img.size(), targetWdivH, cropW, cropH, width, height))
    {
        img(cv::Rect(cropW, cropH, width, height)).copyTo(img);
        //imwrite("AfterCropping.bmp", lastFrame);
    }
}

//opposite to crop image: extend
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
            if (wModulo4 == 1)
                width--;
            if (wModulo4 == 2)
            {
                addW++;
                width -= 2;
            }
            if (wModulo4 == 3)
                width++;
        }

        if (cvBorderType == cv::BORDER_REPLICATE)
        {
            //Camera image on mobile devices have wrongly colored pixels on the right. We want to correct this
            //by cutting away some pixels from the right border in case of BORDER_REPLICATE
            int        numCorrPixRight = 2;
            cv::Rect   roi(0, 0, img.size().width - numCorrPixRight, img.size().height);
            cv::Mat    img2     = img(roi);
            int        addLeft  = addW;
            int        addRight = addW + numCorrPixRight;
            cv::Scalar value(0, 0, 0);
            //BORDER_ISOLATED enables to respect the adjusted roi
            copyMakeBorder(img2, img, addH, addH, addLeft, addRight, cvBorderType | cv::BORDER_ISOLATED, value);

            //Utils::log("extendWithBars", "elapsed time without smooth %f ms", t.elapsedTimeInMilliSec());

            if (addH > 0)
            {
                Utils::log("extendWithBars", "addW blurring not implemented yet!!");
            }
            else if (addW > 0)
            {

                cv::Size iS = img.size();
                //left
                cv::Rect barRoiL(0, 0, addLeft, iS.height);
                cv::Mat  barImgL = img(barRoiL);
                cv::blur(barImgL, barImgL, cv::Size(1, 5), cv::Point(-1, -1));
                //right
                cv::Rect barRoiR(iS.width - addRight, 0, addRight, iS.height);
                cv::Mat  barImgR = img(barRoiR);
                cv::blur(barImgR, barImgR, cv::Size(1, 5), cv::Point(-1, -1));
            }
        }
        else
        {
            cv::Scalar value(0, 0, 0);
            copyMakeBorder(img, img, addH, addH, addW, addW, cvBorderType, value);
        }
    }
    //Utils::log("extendWithBars", "elapsed time total %f ms", t.elapsedTimeInMilliSec());
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

float calcFOVDegFromFocalLengthPix(const float focalLengthPix, const int imgLength)
{
    float fovRad = 2.f * atanf(0.5f * imgLength / focalLengthPix);
    float fovDeg = fovRad * SENS_RAD2DEG;
    return fovDeg;
}
float calcFocalLengthPixFromFOVDeg(const float fovDeg, const int imgLength)
{
    float fovRad = fovDeg * SENS_DEG2RAD;
    float focalLengthPix = 0.5f * imgLength / tanf(0.5f * fovRad);
    return focalLengthPix;
}

};
