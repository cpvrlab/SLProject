//#############################################################################
//  File:      cv06_WarpTriangle.cpp
//  Purpose:   Minimal OpenCV application that warps a triangle into another
//  Copyright: Based on Satya Mallick's Tutorial:
//  https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python
//  Date:      Spring 2018
//#############################################################################

#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
// Warps a triangular regions from img1 to img2
void warpTriangle(Mat&             img1,
                  Mat&             img2,
                  vector<Point2f>& tri1,
                  vector<Point2f>& tri2)
{
    // Find bounding rectangle for each triangle
    Rect rect1 = boundingRect(tri1);
    Rect rect2 = boundingRect(tri2);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> tri1Cropped, tri2Cropped;
    vector<Point>   tri2CroppedInt;
    for (uint i = 0; i < 3; i++)
    {
        tri1Cropped.push_back(Point2f(tri1[i].x - rect1.x, tri1[i].y - rect1.y));
        tri2Cropped.push_back(Point2f(tri2[i].x - rect2.x, tri2[i].y - rect2.y));

        // fillConvexPoly needs a vector of int Point and not Point2f
        tri2CroppedInt.push_back(Point((int)tri2Cropped[i].x, (int)tri2Cropped[i].y));
    }

    // Apply warpImage to small rectangular patches
    Mat img1Cropped;
    img1(rect1).copyTo(img1Cropped);

    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(tri1Cropped, tri2Cropped);

    // Apply the Affine Transform just found to the src image
    Mat img2Cropped = Mat::zeros(rect2.height, rect2.width, img1Cropped.type());
    warpAffine(img1Cropped,
               img2Cropped,
               warpMat,
               img2Cropped.size(),
               INTER_LINEAR,
               BORDER_REFLECT_101);

    // Create white triangle mask
    Mat mask = Mat::zeros(rect2.height, rect2.width, CV_32FC3);
    fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), LINE_AA, 0);

    // Delete all outside of warped triangle
    multiply(img2Cropped, mask, img2Cropped);

    // Delete all inside the target triangle
    multiply(img2(rect2), Scalar(1.0, 1.0, 1.0) - mask, img2(rect2));

    // Add warped triangle to target image
    img2(rect2) = img2(rect2) + img2Cropped;
}
//-----------------------------------------------------------------------------
int main()
{
    std::string projectRoot = std::string(SL_PROJECT_ROOT);

    // Read input image
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    Mat imgIn = imread(projectRoot + "/data/images/textures/Lena.jpg");
    if (imgIn.empty())
    {
        cout << "Could not load image. Is the working dir correct?" << endl;
        exit(1);
    }

    // Convert to float for multiply ops inside warpTriangle
    imgIn.convertTo(imgIn, CV_32FC3, 1 / 255.0);

    // Output image is set to white
    Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
    imgOut     = Scalar(1.0, 1.0, 1.0);

    // Input triangle
    vector<Point2f> triIn;
    triIn.push_back(Point2f(360, 200));
    triIn.push_back(Point2f(60, 250));
    triIn.push_back(Point2f(450, 400));

    // Output triangle
    vector<Point2f> triOut;
    triOut.push_back(Point2f(400, 200));
    triOut.push_back(Point2f(160, 270));
    triOut.push_back(Point2f(400, 400));

    // Warp all pixels inside input triangle to output triangle
    warpTriangle(imgIn, imgOut, triIn, triOut);

    // Draw triangle on the input and output image.

    // Convert back to uint because OpenCV antialiasing
    // does not work on image of type CV_32FC3
    imgIn.convertTo(imgIn, CV_8UC3, 255.0);
    imgOut.convertTo(imgOut, CV_8UC3, 255.0);

    // Draw triangle using this color
    Scalar color = Scalar(255, 150, 0);

    // cv::polylines needs vector of type Point and not Point2f
    vector<Point> triInInt, triOutInt;
    for (uint i = 0; i < 3; i++)
    {
        triInInt.push_back(Point((int)triIn[i].x, (int)triIn[i].y));
        triOutInt.push_back(Point((int)triOut[i].x, (int)triOut[i].y));
    }

    // Draw triangles in input and output images
    polylines(imgIn, triInInt, true, color, 1, LINE_AA);
    polylines(imgOut, triOutInt, true, color, 1, LINE_AA);

    string title1 = "Input";
    imshow(title1, imgIn);
    setWindowProperty(title1, WND_PROP_TOPMOST, 1);

    string title2 = "Output";
    imshow(title2, imgOut);
    setWindowProperty(title2, WND_PROP_TOPMOST, 1);

    waitKey(0);
    return 0;
}
//-----------------------------------------------------------------------------
