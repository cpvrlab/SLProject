//#############################################################################
//  File:      cv07_MeshWarping.cpp
//  Purpose:   Minimal OpenCV application that warps a triangular mesh
//  Copyright: Based on Satya Mallick's Tutorials: https://www.learnopencv.com
//  Date:      Spring 2018
//#############################################################################

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
static void drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size          size = img.size();
    Rect          rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0]   = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1]   = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2]   = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
        }
    }
}
//-----------------------------------------------------------------------------
static void createDelaunay(Mat&                  img,
                           Subdiv2D&             subdiv,
                           vector<Point2f>&      points,
                           bool                  drawAnimated,
                           vector<vector<uint>>& triangleIndexes)
{
    // Insert points into subdiv
    for (Point2f p : points)
    {
        subdiv.insert(p);

        if (drawAnimated)
        {
            Mat img_copy = img.clone();
            drawDelaunay(img_copy, subdiv, Scalar(255, 255, 255));
            imshow("Delaunay Triangulation", img_copy);
            waitKey(100);
        }
    }

    // Unfortunately we don't get the triangles by there original point indexes.
    // We only get them with their vertex coordinates.
    // So we have to map them again to get the triangles with their point indexes.
    Size          size = img.size();
    Rect          rect(0, 0, size.width, size.height);
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<uint>    ind(3);

    for (size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0]   = Point2f(t[0], t[1]);
        pt[1]   = Point2f(t[2], t[3]);
        pt[2]   = Point2f(t[4], t[5]);

        if (rect.contains(pt[0]) &&
            rect.contains(pt[1]) &&
            rect.contains(pt[2]))
        {
            for (uint j = 0; j < 3; j++)
                for (size_t k = 0; k < points.size(); k++)
                    if (abs(pt[j].x - points[k].x) < 1.0 &&
                        abs(pt[j].y - points[k].y) < 1)
                        ind[j] = (uint)k;

            triangleIndexes.push_back(ind);
        }
    }
}
//-----------------------------------------------------------------------------
static void
drawVoronoi(Mat& img, Subdiv2D& subdiv)
{
    vector<vector<Point2f>> facets;
    vector<Point2f>         centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point>         ifacet;
    vector<vector<Point>> ifacets(1);

    for (size_t i = 0; i < facets.size(); i++)
    {
        ifacet.resize(facets[i].size());
        for (size_t j = 0; j < facets[i].size(); j++)
            ifacet[j] = facets[i][j];

        Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;
        fillConvexPoly(img, ifacet, color, 8, 0);

        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, LINE_AA, 0);
        circle(img, centers[i], 2, Scalar(), FILLED, LINE_AA, 0);
    }
}
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
static void warpImage(Mat&                  img1,
                      Mat&                  img2,
                      vector<Point2f>&      points1,
                      vector<Point2f>&      points2,
                      vector<vector<uint>>& triangles)
{
    for (uint i = 0; i < triangles.size(); i++)
    {
        vector<Point2f> tri1;
        tri1.push_back(points1[triangles[i][0]]);
        tri1.push_back(points1[triangles[i][1]]);
        tri1.push_back(points1[triangles[i][2]]);

        vector<Point2f> tri2;
        tri2.push_back(points2[triangles[i][0]]);
        tri2.push_back(points2[triangles[i][1]]);
        tri2.push_back(points2[triangles[i][2]]);

        warpTriangle(img1, img2, tri1, tri2);
    }
}
//-----------------------------------------------------------------------------
int main()
{
    std::string projectRoot = std::string(SL_PROJECT_ROOT);

    // Read input image
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    Mat img_orig = imread(projectRoot + "/data/images/textures/donald_trump.jpg");
    if (img_orig.empty())
    {
        cout << "Could not load image. Is the working dir correct?" << endl;
        return -1;
    }

    // Keep a copy around
    Mat img1 = img_orig.clone();

    // Create a vector of 68 facial landmark points.
    // clang-format off
    vector<Point2f> points = {
      {80, 311}, {80, 357}, {83, 405}, {88, 454}, {96, 502}, {114, 546},
      {144, 580}, {180, 607}, {226, 616}, {278, 611}, {335, 591}, {391, 568},
      {434, 531}, {464, 486}, {479, 433}, {487, 377}, {494, 321}, {109, 259},
      {126, 238}, {154, 235}, {183, 239}, {212, 248}, {283, 255}, {321, 244},
      {359, 240}, {397, 247}, {427, 271}, {241, 298}, {237, 327}, {232, 354},
      {227, 383}, {201, 418}, {215, 423}, {230, 427}, {250, 425}, {270, 421},
      {141, 301}, {159, 293}, {181, 294}, {199, 309}, {178, 313}, {156, 311},
      {309, 312}, {331, 299}, {355, 298}, {376, 309}, {356, 317}, {331, 317},
      {177, 503}, {194, 484}, {213, 473}, {228, 477}, {244, 474}, {271, 488},
      {299, 507}, {271, 523}, {244, 528}, {226, 528}, {209, 525}, {192, 517},
      {190, 500}, {213, 489}, {228, 491}, {244, 491}, {286, 504}, {244, 508},
      {228, 507}, {211, 504}};
    // clang-format on

    // Keep bounding rectangle around face points
    Size    size     = img_orig.size();
    Rect    rectFace = boundingRect(points);
    Point2f center(rectFace.x + rectFace.width * 0.5f,
                   rectFace.y + rectFace.height * 0.5f);

    // Add image border points
    points.push_back(Point2d(0, 0));
    points.push_back(Point2d(size.width / 2, 0));
    points.push_back(Point2d(size.width - 1, 0));
    points.push_back(Point2d(size.width - 1, size.height / 2));
    points.push_back(Point2d(size.width - 1, size.height - 1));
    points.push_back(Point2d(size.width / 2, size.height - 1));
    points.push_back(Point2d(0, size.height - 1));
    points.push_back(Point2d(0, size.height / 2));

    // Create an instance of Subdiv2D
    Rect     rect(0, 0, size.width, size.height);
    Subdiv2D subdiv(rect);

    // Create and draw the Delaunay triangulation
    vector<vector<uint>> triIndexes1;
    createDelaunay(img1, subdiv, points, true, triIndexes1);
    // drawDelaunay(img1, subdiv, Scalar(255, 255, 255));

    // Draw all points red
    for (Point2f p : points)
        circle(img1, p, 3, Scalar(0, 0, 255), FILLED, LINE_AA, 0);

    // Allocate space for voronoi Diagram
    Mat img_voronoi = Mat::zeros(img1.rows, img1.cols, CV_8UC3);

    // Draw voronoi diagram
    drawVoronoi(img_voronoi, subdiv);

    // Show results.
    imshow("Delaunay Triangulation", img1);
    imshow("Voronoi Diagram", img_voronoi);

    // Do head warping with Donald Trumps face
    // Copy the mesh points for warping
    vector<Point2f> wPoints = points;
    float           scale   = 1.0f;
    float           sign    = 1.0;
    img_orig.convertTo(img_orig, CV_32FC3, 1 / 255.0);
    Mat imgW = Mat::ones(img_orig.size(), img_orig.type());

    // Loop scale between 0.8 and 1.2
    while (true)
    {
        scale += sign * 0.01f;
        if (scale >= 1.2f || scale <= 0.8f) sign *= -1.0f;

        // Scale the face points from relative to the face cennter
        for (uint i = 0; i < 68; ++i)
            wPoints[i] = ((points[i] - center) * scale) + center;

        // Warp all triangles
        warpImage(img_orig, imgW, points, wPoints, triIndexes1);
        imshow("Warped Image", imgW);

        // Wait for key to exit loop
        if (waitKey(10) != -1)
            return 0;
    }
}
//-----------------------------------------------------------------------------
