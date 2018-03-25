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
 
int main(int argc,char** argv)
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
        faceDetector.detectMultiScale(gray, faces);

        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        vector<vector<Point2f>> landmarks;

        // Run landmark detector
        bool success = facemark->fit(frame,faces,landmarks);

        if(success)
        {
            // If successful, render the landmarks on the face
            for(int i = 0; i < landmarks.size(); i++)
            {
                for(int j=0; j < landmarks[i].size(); j++)
                    circle(frame, landmarks[i][j], 3, Scalar(0,0,255), -1);
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
