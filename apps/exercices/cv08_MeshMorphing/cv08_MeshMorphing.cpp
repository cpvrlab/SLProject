//#############################################################################
//  File:      cv08_MeshMorphing.cpp
//  Purpose:   Minimal OpenCV application that morphs two triangular meshes
//  Copyright: Based on Satya Mallick's Tutorials: https://www.learnopencv.com
//  Date:      Spring 2018
//#############################################################################

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------
// Draws the Delaunay triangualtion into an image using the Subdiv2D
static void
drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
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
            line(img, pt[0], pt[1], delaunay_color, 2, LINE_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 2, LINE_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 2, LINE_AA, 0);
        }
    }
}
//-----------------------------------------------------------------------------
// Draws the Delaunay triangualtion into an image using the triangle indexes
static void
drawDelaunay(Mat&                  img,
             vector<Point2f>&      points,
             vector<vector<uint>>& triangleIndexes,
             Scalar                delaunay_color)
{
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    for (size_t i = 0; i < triangleIndexes.size(); i++)
    {
        vector<Point2f> tri;
        tri.push_back(points[triangleIndexes[i][0]]);
        tri.push_back(points[triangleIndexes[i][1]]);
        tri.push_back(points[triangleIndexes[i][2]]);

        // Draw rectangles completely inside the image.
        if (rect.contains(tri[0]) && rect.contains(tri[1]) && rect.contains(tri[2]))
        {
            line(img, tri[0], tri[1], delaunay_color, 2, LINE_AA, 0);
            line(img, tri[1], tri[2], delaunay_color, 2, LINE_AA, 0);
            line(img, tri[2], tri[0], delaunay_color, 2, LINE_AA, 0);
        }
    }
}
//-----------------------------------------------------------------------------
static void
createDelaunay(Mat&                  img,
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
// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat&             warpImage,
                          Mat&             src,
                          vector<Point2f>& srcTri,
                          vector<Point2f>& dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform just found to the src image
    warpAffine(src,
               warpImage,
               warpMat,
               warpImage.size(),
               INTER_LINEAR,
               BORDER_REFLECT_101);
}
//-----------------------------------------------------------------------------
// Warps and alpha blends two triangles from img1 and img2 into imgM
void morphTriangle(Mat&             img1,
                   Mat&             img2,
                   Mat&             imgM,
                   vector<Point2f>& t1,
                   vector<Point2f>& t2,
                   vector<Point2f>& tM,
                   float            alpha)
{
    // Find bounding rectangle for each triangle
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    Rect rM = boundingRect(tM);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1RectFlt, t2RectFlt, tMRectFlt;
    vector<Point2i> tMRectInt; // for fillConvexPoly we need ints
    for (uint i = 0; i < 3; i++)
    {
        tMRectFlt.push_back(Point2f(tM[i].x - rM.x, tM[i].y - rM.y));
        tMRectInt.push_back(Point2i((int)(tM[i].x - rM.x), (int)(tM[i].y - rM.y)));
        t1RectFlt.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
        t2RectFlt.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
    }

    // Create white triangle mask
    Mat mask = Mat::zeros(rM.height, rM.width, CV_32FC3);
    fillConvexPoly(mask, tMRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);

    Mat warpImage1 = Mat::zeros(rM.height, rM.width, img1Rect.type());
    Mat warpImage2 = Mat::zeros(rM.height, rM.width, img2Rect.type());

    applyAffineTransform(warpImage1, img1Rect, t1RectFlt, tMRectFlt);
    applyAffineTransform(warpImage2, img2Rect, t2RectFlt, tMRectFlt);

    // Alpha blend rectangular patches into new image
    Mat imgRect = (1.0f - alpha) * warpImage1 + alpha * warpImage2;

    // Delete all outside of triangle
    multiply(imgRect, mask, imgRect);

    // Delete all inside the target triangle
    multiply(imgM(rM), Scalar(1.0f, 1.0f, 1.0f) - mask, imgM(rM));

    // Add morphed triangle to target image
    imgM(rM) = imgM(rM) + imgRect;
}
//-----------------------------------------------------------------------------
int main()
{
    std::string projectRoot = std::string(SL_PROJECT_ROOT);

    // Read input images
    Mat img1 = imread(projectRoot + "/data/images/textures/hillary_clinton.jpg");
    Mat img2 = imread(projectRoot + "/data/images/textures/donald_trump.jpg");
    if (img1.empty() || img2.empty())
    {
        cout << "Could not load image. Is the working dir correct?" << endl;
        return -1;
    }

    // convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);

    // clang-format off
    //Read points of face 1
    vector<Point2f> points1 = {
        {125, 358}, {128, 402}, {132, 445}, {137, 490}, {151, 532}, {178, 566},
        {216, 595}, {260, 616}, {304, 622}, {347, 612}, {388, 591}, {426, 563},
        {452, 526}, {466, 482}, {470, 437}, {474, 392}, {477, 345}, {150, 332},
        {171, 312}, {200, 304}, {230, 307}, {259, 319}, {315, 314}, {345, 299},
        {377, 294}, {410, 300}, {434, 319}, {289, 350}, {290, 382}, {291, 413},
        {292, 444}, {258, 458}, {275, 462}, {294, 467}, {313, 460}, {331, 454},
        {184, 358}, {201, 344}, {224, 345}, {245, 363}, {224, 368}, {201, 368},
        {339, 358}, {358, 337}, {381, 335}, {401, 349}, {383, 359}, {360, 361},
        {214, 493}, {245, 489}, {274, 488}, {295, 489}, {316, 485}, {346, 483},
        {381, 484}, {351, 524}, {321, 540}, {299, 543}, {277, 542}, {246, 530},
        {223, 495}, {275, 499}, {296, 499}, {317, 496}, {372, 487}, {319, 523},
        {298, 526}, {276, 525}, {495, 400}, {264, 736}, {0, 774}, {599, 706},
        {0, 0}, {0, 400}, {0, 799}, {300, 799}, {599, 799}, {599, 400}, {599, 0},
        {300, 0}};

    //Read points of face 2
    vector<Point2f> points2 = {
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
        {228, 507}, {211, 504}, {538, 410}, {215, 727}, {0, 760}, {599, 597},
        {0, 0}, {0, 400}, {0, 799}, {300, 799}, {599, 799}, {599, 400}, {599, 0},
        {300, 0}};
    // clang-format on

    // Create an instance of Subdiv2D
    Rect     rect(0, 0, img1.size().width, img1.size().height);
    Subdiv2D subdiv(rect);

    // Create and draw the Delaunay triangulation
    vector<vector<uint>> triangleIndexes;
    createDelaunay(img1, subdiv, points1, false, triangleIndexes);

    // Draw the Delaunay triangulation of face 1
    Mat img1D = img1.clone();
    drawDelaunay(img1D, subdiv, Scalar(255, 255, 255));
    imshow("Delaunay Triangulation of Face 1", img1D / 255.0f);

    // Draw the Delaunay triangulation of face 1
    Mat img2D = img2.clone();
    drawDelaunay(img2D, points2, triangleIndexes, Scalar(255, 255, 255));
    imshow("Delaunay Triangulation of Face 2", img2D / 255.0f);

    // Loop blend factor alpha between 0 and 1
    float alpha = 0.5f;
    float sign  = 1.0f;

    while (true)
    {
        alpha += sign * 0.05f;
        if (alpha >= 1.0f || alpha <= 0.0f) sign *= -1.0f;

        // compute weighted average point coordinates
        vector<Point2f> pointsM;
        for (uint i = 0; i < points1.size(); i++)
        {
            float x = (1 - alpha) * points1[i].x + alpha * points2[i].x;
            float y = (1 - alpha) * points1[i].y + alpha * points2[i].y;
            pointsM.push_back(Point2f(x, y));
        }

        // empty image for morphed face
        Mat imgM = Mat::zeros(img1.size(), CV_32FC3);

        // Loop over all triangles and morph them
        for (size_t i = 0; i < triangleIndexes.size(); i++)
        {
            vector<Point2f> t1, t2, tM;
            t1.push_back(points1[triangleIndexes[i][0]]);
            t1.push_back(points1[triangleIndexes[i][1]]);
            t1.push_back(points1[triangleIndexes[i][2]]);
            t2.push_back(points2[triangleIndexes[i][0]]);
            t2.push_back(points2[triangleIndexes[i][1]]);
            t2.push_back(points2[triangleIndexes[i][2]]);
            tM.push_back(pointsM[triangleIndexes[i][0]]);
            tM.push_back(pointsM[triangleIndexes[i][1]]);
            tM.push_back(pointsM[triangleIndexes[i][2]]);

            morphTriangle(img1, img2, imgM, t1, t2, tM, alpha);
        }

        imshow("Morphed Face", imgM / 255.0);

        // Wait for key to exit loop
        if (waitKey(10) != -1) return 0;
    }
}
//-----------------------------------------------------------------------------
