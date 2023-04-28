//#############################################################################
//  File:      cv01_ChangeBrightnessAndContrast.cpp
//  Purpose:   Minimal OpenCV application that changes brightness and contrast
//             Taken from the OpenCV Tutorial:
//             http://docs.opencv.org/3.1.0/d3/dc1/tutorial_basic_linear_transform.html
//             See also the root page of all OpenCV Tutorials:
//             http://docs.opencv.org/3.1.0/d9/df8/tutorial_root.html
//  Date:      Spring 2017
//#############################################################################

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    std::string projectRoot = std::string(SL_PROJECT_ROOT);

    // Read input image
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    Mat img = imread(projectRoot + "/data/images/textures/Lena.jpg");
    if (img.empty())
    {
        cout << "Could not load img. Is the working dir correct?" << endl;
        exit(1);
    }

    // Show original image
    string title = "Original image";
    imshow(title, img);
    setWindowProperty(title, WND_PROP_TOPMOST, 1);

    double contrast   = 2.0;
    int    brightness = 50;

    ////////////////////////////////
    // Method 1: Do it pixel wise //
    ////////////////////////////////

    // Do the operation new_img(i,j) = contrast * img(i,j) + brightness pixel wise
    Mat img1 = Mat::zeros(img.size(), img.type());
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                img1.at<Vec3b>(y, x)[c] =
                  saturate_cast<uchar>(contrast * (img.at<Vec3b>(y, x)[c]) + brightness);
            }
        }
    }
    string title1 = "img1.at<Vec3b>(y,x)[c] = contrast*(img.at<Vec3b>(y,x)[c]) + brightness";
    imshow(title1, img1);
    setWindowProperty(title1, WND_PROP_TOPMOST, 1);

    ////////////////////////////////////
    // Method 2: Do it with convertTo //
    ////////////////////////////////////

    Mat img2 = Mat::zeros(img.size(), img.type());
    img.convertTo(img2, -1, contrast, brightness);
    string title2 = "img.convertTo(img2, -1, contrast, brightness);";
    imshow(title2, img2);
    setWindowProperty(title2, WND_PROP_TOPMOST, 1);

    //////////////////////////////////////////
    // Method 3: Using overloaded operators //
    //////////////////////////////////////////

    Mat    img3   = contrast * img + brightness;
    string title3 = "img3 = contrast * img + brightness";
    imshow(title3, img3);
    setWindowProperty(title3, WND_PROP_TOPMOST, 1);

    // Wait until user presses some key
    waitKey(0);
    return 0;
}
