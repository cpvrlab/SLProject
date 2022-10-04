//#############################################################################
//  File:      cv15_YoloV3.cpp
//  Purpose:   Object Detection with YOLO V3 and OpenCV
//  Taken from Satya Mallic on: http://www.learnopencv.com
//  Date:      December 2021
//#############################################################################

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// Initialize the parameters
string         projectRoot   = string(SL_PROJECT_ROOT); //!< Directory of executable
float          confThreshold = 0.5f;                    // Confidence threshold
float          nmsThreshold  = 0.4f;                    // Non-maximum suppression threshold
int            inpWidth      = 416;                     // Width of network's input image
int            inpHeight     = 416;                     // Height of network's input image
vector<string> classes;
const char*    keys =
  "{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
  "{image i        |<none>| input image   }"
  "{video v        |<none>| input video   }"
  "{device d       |<cpu>| input device   }";

//-----------------------------------------------------------------------------
// Helper function that checks if a file exists
bool fileExists(const std::string& name)
{
    ifstream f(name.c_str());
    return f.good();
}
//-----------------------------------------------------------------------------
// Get the names of the output layers of the YOLO V3 network
vector<String> getOutputsNames(const dnn::Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        // Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        // get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
//-----------------------------------------------------------------------------
// Draw the predicted bounding box
void drawPred(int   classId,
              float conf,
              int   left,
              int   top,
              int   right,
              int   bottom,
              Mat&  frame)
{
    // Draw a rectangle displaying the bounding box
    rectangle(frame,
              Point(left, top),
              Point(right, bottom),
              Scalar(255, 178, 50),
              3);

    // Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    // Display the label at the top of the bounding box
    int  baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top            = max(top, labelSize.height);
    rectangle(frame,
              Point(left, top - (int)round(1.5 * labelSize.height)),
              Point(left + (int)round(1.5 * labelSize.width), top + baseLine),
              Scalar(255, 255, 255),
              FILLED);
    putText(frame,
            label,
            Point(left, top),
            FONT_HERSHEY_SIMPLEX,
            0.75,
            Scalar(0, 0, 0),
            1);
}
//-----------------------------------------------------------------------------
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int>   classIds;
    vector<float> confidences;
    vector<Rect>  boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat    scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point  classIdPoint;
            double confidence;

            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width   = (int)(data[2] * frame.cols);
                int height  = (int)(data[3] * frame.rows);
                int left    = centerX - width / 2;
                int top     = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int  idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx],
                 confidences[idx],
                 box.x,
                 box.y,
                 box.x + box.width,
                 box.y + box.height,
                 frame);
    }
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Load names of classes
    string   classesFile = projectRoot + "/data/opencv/yolov3/coco.names";
    ifstream ifs(classesFile.c_str());
    string   line;
    while (getline(ifs, line)) classes.push_back(line);

    string device = "cpu";
    device        = parser.get<String>("device");

    // Give the configuration and weight files for the model
    string modelConfiguration = projectRoot + "/data/opencv/yolov3/yolov3.cfg";
    string modelWeights       = projectRoot + "/data/opencv/yolov3/yolov3.weights";
    if (!fileExists(modelWeights))
    {
        cout << "Please download first the Yolo V3 model weights from" << endl;
        cout << "https://pjreddie.com/media/files/yolov3.weights" << endl;
        cout << "into /data/opencv/yolov3/" << endl;
        exit(1);
    }

    // Load the network
    dnn::Net net = dnn::readNetFromDarknet(modelConfiguration, modelWeights);

    if (device == "cpu")
    {
        cout << "Using CPU device" << endl;
        net.setPreferableBackend(dnn::DNN_TARGET_CPU);
    }
    else if (device == "gpu")
    {
        cout << "Using GPU device" << endl;
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    }

    // Open a video file or an image file or a camera stream.
    string       str, outputFile;
    VideoCapture cap;
    VideoWriter  video;
    Mat          frame, blob;

    try
    {

        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<String>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcam
        else
        {
            cap.open(0);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        }
    }
    catch (...)
    {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }

    // Get the video writer initialized to save the output video
    if (!parser.has("image") && parser.has("video"))
    {
        video.open(outputFile,
                   VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   28,
                   Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                        (int)cap.get(CAP_PROP_FRAME_HEIGHT)));
    }

    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    cv::namedWindow(kWinName, WINDOW_NORMAL);
    cv::resizeWindow(kWinName, 1280, 720);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty())
        {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        dnn::blobFromImage(frame,
                           blob,
                           1 / 255.0,
                           cv::Size(inpWidth, inpHeight),
                           Scalar(0, 0, 0),
                           true,
                           false);

        // Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        // Remove the bounding boxes with low confidence
        postprocess(frame, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double         freq  = getTickFrequency() / 1000;
        double         t     = net.getPerfProfile(layersTimes) / freq;
        string         label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        if (parser.has("image"))
            imwrite(outputFile, detectedFrame);
        else if (parser.has("video"))
            video.write(detectedFrame);

        imshow(kWinName, frame);
    }

    cap.release();
    if (!parser.has("image")) video.release();

    return 0;
}
//-----------------------------------------------------------------------------