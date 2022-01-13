//#############################################################################
//  File:      cv02_CalderonFilter.cpp
//  Purpose:   Minimal OpenCV app for the Instagram Calderon filter
//             Taken from the OpenCV Tutorial at https://www.learnopencv.com/
//             from Satja Malick
//  Date:      Spring 2018
//#############################################################################

#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//----------------------------------------------------------------------------
// Piecewise linear interpolation implemented on a particular Channel
void interpolation(uchar* lut,
                   float* fullRange,
                   float* curve,
                   float* originalVal)
{
    for (int i = 0; i < 256; i++)
    {
        int   j = 0;
        float a = fullRange[i];

        while (a > originalVal[j]) j++;

        if (a == originalVal[j])
        {
            lut[i] = (uchar)curve[j];
            continue;
        }

        float slope    = (curve[j] - curve[j - 1]) / (originalVal[j] - originalVal[j - 1]);
        float constant = curve[j] - slope * originalVal[j];
        lut[i]         = (uchar)(slope * fullRange[i] + constant);
    }
}
//----------------------------------------------------------------------------
Mat clarendon(Mat original)
{
    // Enhance the channel for any image BGR or HSV etc
    Mat   img      = original.clone();
    float origin[] = {0, 28, 56, 85, 113, 141, 170, 198, 227, 255};
    float rCurve[] = {0, 16, 35, 64, 117, 163, 200, 222, 237, 249};
    float gCurve[] = {0, 24, 49, 98, 141, 174, 201, 223, 239, 255};
    float bCurve[] = {0, 38, 66, 104, 139, 175, 206, 226, 245, 255};

    // Splitting the channels
    vector<Mat> channels(3);
    split(img, channels);

    // Create a LookUp Table
    float fullRange[256];
    for (int i = 0; i < 256; i++)
        fullRange[i] = (float)i;

    Mat    lookUpTable(1, 256, CV_8U);
    uchar* lut = lookUpTable.ptr();

    // Create lookup table and apply it on blue channel
    interpolation(lut, fullRange, bCurve, origin);
    LUT(channels[0], lookUpTable, channels[0]);

    // Create lookup table and apply it on green channel
    interpolation(lut, fullRange, gCurve, origin);
    LUT(channels[1], lookUpTable, channels[1]);

    // Create lookup table and apply it on red channel
    interpolation(lut, fullRange, rCurve, origin);
    LUT(channels[2], lookUpTable, channels[2]);

    // Merge the channels
    Mat output;
    merge(channels, output);

    return output;
}
//----------------------------------------------------------------------------
int main()
{
    std::string projectRoot = std::string(SL_PROJECT_ROOT);

    // Read input image
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    Mat image = imread(projectRoot + "/data/images/textures/girl.jpg");
    if (image.empty())
    {
        cout << "Could not load img. Is the working dir correct?" << endl;
        exit(1);
    }

    //////////////////////////////
    Mat output = clarendon(image);
    //////////////////////////////

    string title1 = "Original Image";
    imshow(title1, image);
    setWindowProperty(title1, WND_PROP_TOPMOST, 1);

    string title2 = "Calderon Image";
    imshow(title2, output);
    setWindowProperty(title2, WND_PROP_TOPMOST, 1);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
//----------------------------------------------------------------------------
