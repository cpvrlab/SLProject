//#############################################################################
//  File:      cv13_FacialLandmarkDetection.cpp
//  Purpose:   Minimal OpenCV app for facial landmark detection without dlib
//  Taken from Satya Mallic on: http://www.learnopencv.com
//  Date:      Authumn 2017
//#############################################################################

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
 
using namespace std;
using namespace cv;
using namespace cv::face;

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
        // Find face
        vector<Rect> faces;
        // Convert frame to grayscale because
        // faceDetector requires grayscale image.
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
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

        if(success)
        {
            for(int i=0; i < landmarks.size(); i++)
            {
                rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);

                for(int j=0; j < landmarks[i].size(); j++)
                    circle(frame, landmarks[i][j], 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Display results
        imshow("Facial Landmark Detection", frame);
        
        // Wait for key to exit loop
        if (waitKey(10) != -1)
            return 0;
    }
    return 0;
}
//-----------------------------------------------------------------------------
