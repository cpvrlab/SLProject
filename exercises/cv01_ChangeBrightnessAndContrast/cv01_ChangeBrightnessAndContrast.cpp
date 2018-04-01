//#############################################################################
//  File:      cv01_ChangeBrightnessAndContrast.cpp
//  Purpose:   Minimal OpenCV application that changes brightness and contrast
//             Taken from the OpenCV Tutorial:
//             http://docs.opencv.org/3.1.0/d3/dc1/tutorial_basic_linear_transform.html
//             See also the root page of all OpenCV Tutorials:
//             http://docs.opencv.org/3.1.0/d9/df8/tutorial_root.html
//  Date:      Spring 2017
//#############################################################################

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // Read input image
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    Mat image = imread("../_data/images/textures/Lena.tiff");
    if (image.empty())
    {   cout << "Could not load image. Is the working dir correct?" << endl;
        exit(1);
    }

    Mat new_image = Mat::zeros(image.size(), image.type() );

    double alpha = 2.0;
    int beta = 50;

    // Do the operation new_image(i,j) = alpha * image(i,j) + beta
    // The following loop does the same as: 
    // image.convertTo(new_image, -1, alpha, beta);
    for( int y = 0; y < image.rows; y++ ) 
    {
        for( int x = 0; x < image.cols; x++ ) 
        {
            for( int c = 0; c < 3; c++ ) 
            {
                new_image.at<Vec3b>(y,x)[c] =
                saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }

    // Create Windows
    namedWindow("Original Image", 1);
    namedWindow("New Image", 1);

    // Show stuff
    imshow("Original Image", image);
    imshow("New Image", new_image);

    // Wait until user presses some key
    waitKey(0);
    return 0;
}
