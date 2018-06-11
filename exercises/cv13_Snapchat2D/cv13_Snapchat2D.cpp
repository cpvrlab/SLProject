//#############################################################################
//  File:      cv13_Snapchat2D.cpp
//  Purpose:   Minimal OpenCV app for a 2D Snapchatfilter
//  Taken from Satya Mallic on: http://www.learnopencv.com
//  Date:      Authumn 2017
//#############################################################################

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

//-----------------------------------------------------------------------------
static void drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);

    for(size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {   line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}
//-----------------------------------------------------------------------------
static void createDelaunay(Mat& img,
                           Subdiv2D& subdiv,
                           vector<Point2f>& points,
                           bool drawAnimated,
                           vector<vector<int>>& triangleIndexes)
{
    // Insert points into subdiv
    for (Point2f p : points)
    {
        subdiv.insert(p);

        if (drawAnimated)
        {   Mat img_copy = img.clone();
            drawDelaunay(img_copy, subdiv, Scalar(255, 255, 255));
            imshow("Delaunay Triangulation", img_copy);
            waitKey(100);
        }
    }

    // Unfortunately we don't get the triangles by there original point indexes.
    // We only get them with their vertex coordinates.
    // So we have to map them again to get the triangles with their point indexes.
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5 ]);

        if (rect.contains(pt[0]) &&
            rect.contains(pt[1]) &&
            rect.contains(pt[2]))
        {
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 &&
                       abs(pt[j].y - points[k].y) < 1)
                        ind[j] = (int)k;

            triangleIndexes.push_back(ind);
        }
    }
}
//-----------------------------------------------------------------------------
// Warps a triangular regions from img1 to img2
void warpTriangle(Mat &img1,
                   Mat &img2,
                   vector<Point2f>& tri1,
                   vector<Point2f>& tri2)
{
    // Find bounding rectangle for each triangle
    Rect rect1 = boundingRect(tri1);
    Rect rect2 = boundingRect(tri2);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> tri1Cropped, tri2Cropped;
    vector<Point> tri2CroppedInt;
    for(int i = 0; i < 3; i++)
    {
        tri1Cropped.push_back(Point2f(tri1[i].x-rect1.x, tri1[i].y-rect1.y));
        tri2Cropped.push_back(Point2f(tri2[i].x-rect2.x, tri2[i].y-rect2.y));

        // fillConvexPoly needs a vector of int Point and not Point2f
        tri2CroppedInt.push_back(Point((int)tri2Cropped[i].x,(int)tri2Cropped[i].y));
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
    multiply(img2(rect2), Scalar(1.0,1.0,1.0) - mask, img2(rect2));

    // Add warped triangle to target image
    img2(rect2) = img2(rect2) + img2Cropped;
}
//-----------------------------------------------------------------------------
static void warpImage(Mat& img1,
                      Mat& img2,
                      vector<Point2f>& points1,
                      vector<Point2f>& points2,
                      vector<vector<int>>& triangles)
{
    for(size_t i = 0; i < triangles.size(); i++)
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
    // Load Face Detector
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging
    CascadeClassifier faceDetector("../_data/opencv/haarcascades/haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("../_data/calibrations/lbfmodel.yaml");

    // Set up webcam for video capture
    VideoCapture cam(0);

    // Variable to store a video frame and its grayscale
    Mat frame, gray;

    // Read a frame
    while(cam.read(frame))
    {
        // Convert frame to grayscale because faceDetector requires grayscale image
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        int min = (int)(frame.rows*0.4f); // the bigger min the faster
        int max = (int)(frame.rows*0.8f); // the smaller max the faster
        cv::Size minSize(min, min);
        cv::Size maxSize(max, max);
        faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, minSize, maxSize);

        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        vector<vector<Point2f>> landmarks;

        // Run landmark detector
        bool success = facemark->fit(frame,faces,landmarks);

        if(success && landmarks.size() >= 1)
        {
            // Add image border points at the end of the landmarks vector
            Size size = frame.size();
            landmarks[0].push_back(Point2d(0,0));
            landmarks[0].push_back(Point2d(size.width/2,0));
            landmarks[0].push_back(Point2d(size.width-1,0));
            landmarks[0].push_back(Point2d(size.width-1,size.height/2));
            landmarks[0].push_back(Point2d(size.width-1,size.height-1));
            landmarks[0].push_back(Point2d(size.width/2,size.height-1));
            landmarks[0].push_back(Point2d(0,size.height-1));
            landmarks[0].push_back(Point2d(0,size.height/2));

            // Create an instance of Subdiv2D
            Rect rect(0, 0, size.width, size.height);
            Subdiv2D subdiv(rect);

            // Create and draw the Delaunay triangulation
            vector<vector<int>> triIndexes1;
            createDelaunay(frame, subdiv, landmarks[0], false, triIndexes1);
            drawDelaunay(frame, subdiv, Scalar(255, 255, 255));

            /////////////////////////////////////////////////////////////
            // Warp some triangles with some points you want to change //
            /////////////////////////////////////////////////////////////

            // ???

            // Display results
            imshow("Snapchat Warp Filter", frame);
        }


        // Wait for key to exit loop
        if (waitKey(10) != -1)
            return 0;
    }
    return 0;
}
//-----------------------------------------------------------------------------
