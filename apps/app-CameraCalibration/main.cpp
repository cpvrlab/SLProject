#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Utils.h>
#include "CamCalibration.h"
#include "CamCalibrationManager.h"

//-----------------------------------------------------------------------------
// global variables:
std::unique_ptr<cv::VideoCapture> cap;
//id of live camera stream (most of the time it is 0 or 1)
int captureId = -1;
//full path of video file
std::string videoFileName;
//full path of directory contining pictures of chessboard
std::string pictureDir;
//if videoLoops == 1 the video is repeated endlessly
bool videoLoop            = false;
bool helpRequired         = false;
bool designMeAChessboard  = false;
bool showExtractionResult = false;
//bool calibFixAspectRatio        = false;
//bool calibZeroTangentDistortion = false;
//bool calibFixPrincipalPoint     = false;

//cv::Size2i defaultCamResolution = cv::Size2i(640, 480);
cv::Size2i  defaultCamResolution = cv::Size2i(1920, 1080);
std::string windowName           = "Camera Calibration";
cv::Size    chessboardSize(8, 5);
//cv::Size chessboardSize(10, 7);
float squareLength = 0.168f;
//float squareLength = 20.4f / 6.f;
int defaultNumberOfPictures = 30;
//cv::Size chessboardSize(8, 5);
bool useReleaseObjectsMethod = true;

std::unique_ptr<CamCalibrationManager> calibMgr;
bool                                   calibrated = false;

//-----------------------------------------------------------------------------
//! Draws text to image that informs about current status
void drawStatusMessage(const cv::Mat& img)
{
    //print help
    cv::putText(img, calibMgr->getHelpMsg(), cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0));
    //print status text
    cv::putText(img, calibMgr->getStatusMsg(), cv::Point(30, 60), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0));
}
//-----------------------------------------------------------------------------
void extractCorners(cv::Mat frame, std::string savePath)
{
    static int               iImg = 0;
    std::vector<cv::Point2f> corners;
    bool                     foundCorners = cv::findChessboardCorners(frame, chessboardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
    if (foundCorners)
    {
        std::cout << "corners found" << std::endl;
        cv::Mat imgGray;
        cv::cvtColor(frame, imgGray, cv::COLOR_BGR2GRAY);
        cornerSubPix(imgGray, corners, cv::Size(31, 31), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.0001));
        //copy corners to image collection
        calibMgr->addCorners(corners);

        std::stringstream ss;
        ss << savePath << "/img" << iImg << ".png";
        cv::imwrite(ss.str(), frame);
        cv::drawChessboardCorners(frame, chessboardSize, cv::Mat(corners), foundCorners);
        std::stringstream sss;
        sss << savePath << "/eval" << iImg << ".png";
        cv::imwrite(sss.str(), frame);
        iImg++;

        //simulate a snapshot
        if (showExtractionResult)
        {
            cv::bitwise_not(frame, frame);
            drawStatusMessage(frame);
            cv::imshow(windowName, frame);
            cv::waitKey(200);
        }
    }
    else
    {
        std::cout << "corners NOT found" << std::endl;
        cv::drawChessboardCorners(frame, chessboardSize, cv::Mat(corners), foundCorners);
    }
}
//-----------------------------------------------------------------------------
void onMouseCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN:
            //get frame
            if (cap->isOpened())
            {
                cv::Mat frame;
                *cap >> frame;

                if (frame.empty())
                    break;

                std::string savePath = Utils::getCurrentWorkingDir() + "test_data/";
                if (!Utils::fileExists(savePath))
                    Utils::makeDir(savePath);
                extractCorners(frame, savePath);

                if (calibMgr->readyForCalibration())
                {
                    CVCalibration calib = calibMgr->calculateCalibration(
                      false, false, false, false, false, false);
                    std::cout << "camera matrix: " << calib.cameraMat() << std::endl;
                    std::cout << "distortion coefficients: " << calib.distortion() << std::endl;
                    //todo
                    calib.save("", "CameraCalibration0.xml");
                    calibrated = true;
                }
            }
            break;
    }
}
//-----------------------------------------------------------------------------
void designChessboard(int ncols, int nrows)
{
    std::cout << "designing chessboard" << std::endl;
    //chessboard designer
    //int ncols = 16;
    //int nrows = 9;
    int pixPerRect = 100;
    //black rect img
    cv::Mat white(100, 100, cv::DataType<unsigned char>::type, 255);
    cv::Mat black = cv::Mat::zeros(100, 100, cv::DataType<unsigned char>::type);
    //cv::imwrite("blackrect.png", black);
    //cv::imwrite("whiterect.png", white);

    cv::Mat row0 = white.clone();
    cv::Mat row1 = black.clone();
    for (int col = 0; col < (ncols / 2); ++col)
    {
        cv::hconcat(row0, black, row0);
        cv::hconcat(row0, white, row0);
        cv::hconcat(row1, white, row1);
        cv::hconcat(row1, black, row1);
    }
    //cv::imwrite("row0.png", row0);
    //cv::imwrite("row1.png", row1);

    cv::Mat twoRows;
    cv::vconcat(row0, row1, twoRows);
    cv::Mat chessBoard = twoRows.clone();
    for (int row = 0; row < nrows / 2; ++row)
    {
        cv::vconcat(chessBoard, twoRows, chessBoard);
    }
    cv::imwrite("chessboard.png", chessBoard);
}

//-----------------------------------------------------------------------------
void printHelp()
{
    std::stringstream ss;
    ss << "Usage: camera_calibration_app.exe [Options] or ./camera_calibration_app [Options]" << std::endl;
    ss << "Example1 (win):  camera_calibration_app.exe -captureId 0" << std::endl;
    ss << "Example2 (unix): ./camera_calibration_app -videoFileName data/video/test.mp4 -videoLoops 1" << std::endl;
    ss << "" << std::endl;
    ss << "Options: " << std::endl;
    ss << "  -h/-help           print this help, e.g. -h" << std::endl;
    ss << "  -captureId         id of webcam stream (most of the time it is 0 or 1), e.g. -captureId 0" << std::endl;
    ss << "  -camResolution     camera resolution for live video stream (default: " << defaultCamResolution << "), e.g. -camResolution 640 480" << std::endl;
    ss << "  -videoFileName     path to video file, e.g. -videoFileName C:/calibImgs/calibVideo.mp4" << std::endl;
    ss << "  -videoLoop         continue video from beginning when finished (default: false, transferring -videoLoop leads to true)" << std::endl;
    ss << "  -numOfPictures     number of picture to capture from video (default: " << defaultNumberOfPictures << ", e.g. -numOfPictures 20" << std::endl;

    ss << "  -pictureDir         path to directory containing images of chessboard, e.g. -pictureDir C:/calibImgs" << std::endl;

    //ss << "  -fixPrincipalPoint     sets flag cv::CALIB_FIX_PRINCIPAL_POINT (default: false, transferring -fixPrincipalPoint leads to true)" << std::endl;
    //ss << "  -zeroTangentDistortion sets flag cv::CALIB_ZERO_TANGENT_DIST (default: false, transferring -zeroTangentDistortion leads to true)" << std::endl;
    //ss << "  -fixAspectRatio        sets flag cv::CALIB_FIX_ASPECT_RATIO (default: false, transferring -fixAspectRatio leads to true)" << std::endl;

    ss << "  -chessboardSize    number of inner corners of chessboard used for chessboard designer and chessboard identification in any case. Width has to be equal sized height unequal (default: " << chessboardSize << "), e.g. -chessboardSize 16 9" << std::endl;
    ss << "  -designChessboard  designs chessboard depending on chessboardSize (inner corners) and stores it beside executable as chessboard.png, e.g. -designChessboard" << std::endl;
    ss << "" << std::endl;
    ss << "During runtime: " << std::endl;
    ss << "  Press 'q' with focus on video dialog to stop execution" << std::endl;
    ss << "  Not all parameter combinations make sense together." << std::endl;

    std::cout << ss.str() << std::endl;
}
//-----------------------------------------------------------------------------
void readArgs(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "-captureId"))
        {
            std::istringstream stream(argv[++i]);
            stream >> captureId;
            std::cout << "capture id: " << captureId << std::endl;
        }
        else if (!strcmp(argv[i], "-pictureDir"))
        {
            pictureDir = argv[++i];
        }
        else if (!strcmp(argv[i], "-videoFileName"))
        {
            videoFileName = argv[++i];
        }
        else if (!strcmp(argv[i], "-videoLoop"))
        {
            videoLoop = true;
        }
        else if (!strcmp(argv[i], "-designChessboard"))
        {
            designMeAChessboard = true;
        }
        //else if (!strcmp(argv[i], "-zeroTangentDistortion"))
        //{
        //    calibZeroTangentDistortion = true;
        //}
        //else if (!strcmp(argv[i], "-fixAspectRatio"))
        //{
        //    calibFixAspectRatio = true;
        //}
        //else if (!strcmp(argv[i], "-fixPrincipalPoint"))
        //{
        //    calibFixPrincipalPoint = true;
        //}
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help"))
        {
            helpRequired = true;
        }
        else if (!strcmp(argv[i], "-camResolution"))
        {
            defaultCamResolution.width  = std::atoi(argv[++i]);
            defaultCamResolution.height = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-chessboardSize"))
        {
            chessboardSize.width  = std::atoi(argv[++i]);
            chessboardSize.height = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-numOfPictures"))
        {
            defaultNumberOfPictures = std::atoi(argv[++i]);
        }
    }
}
//-----------------------------------------------------------------------------
bool initCaptureTool()
{
    if (captureId != -1)
    {
        cap = std::make_unique<cv::VideoCapture>(captureId);
    }
    else if (videoFileName.size())
    {
        cap = std::make_unique<cv::VideoCapture>(videoFileName);
    }

    if (!cap || !cap->isOpened())
    {
        std::cout << "ERROR: you have to provide an capture id or video file path as options!" << std::endl;
        printHelp();
        return false;
    }

    if (captureId != -1)
    {
        // fsb1: does not work for me:
        cap->set(cv::CAP_PROP_FRAME_WIDTH, defaultCamResolution.width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, defaultCamResolution.height);
        cap->set(cv::CAP_PROP_AUTOFOCUS, 1);
    }
    return true;
}
//-----------------------------------------------------------------------------
void calibAndSave(bool        fixAspectRatio,
                  bool        zeroTangentDistortion,
                  bool        fixPrincipalPoint,
                  bool        calibRationalModel,
                  bool        calibTiltedModel,
                  bool        calibThinPrismModel,
                  std::string outputDir,
                  int         i,
                  ofstream&   file)
{
    CVCalibration calib = calibMgr->calculateCalibration(
      fixAspectRatio, zeroTangentDistortion, fixPrincipalPoint, calibRationalModel, calibTiltedModel, calibThinPrismModel);
    std::cout << "camera matrix: " << calib.cameraMat() << std::endl;
    std::cout << "distortion coefficients: " << calib.distortion() << std::endl;
    calib.save(outputDir, "CameraCalibration" + std::to_string(i) + ".xml");
    //file << "fixAspectRatio;zeroTangentDistortion;fixPrincipalPoint;calibRationalModel;calibTiltedModel;calibThinPrismModel;fx;fy;cx;cy;k1;k2;p1;p2;k3;k4;k5;k6;s1;s2;s3;s4;tauX;tauY;reprojError\n";
    cv::Mat distortion = calib.distortion();
    file << std::fixed << std::setprecision(9)
         << fixAspectRatio << ","
         << zeroTangentDistortion << ","
         << fixPrincipalPoint << ","
         << calibRationalModel << ","
         << calibTiltedModel << ","
         << calibThinPrismModel << ","

         << calib.fx() << ","
         << calib.fy() << ","
         << calib.cx() << ","
         << calib.cy() << ","

         << calib.k1() << ","
         << calib.k2() << ","
         << calib.p1() << ","
         << calib.p2() << ","

         << calib.k3() << ","
         << calib.k4() << ","
         << calib.k5() << ","
         << calib.k6() << ","

         << calib.s1() << ","
         << calib.s2() << ","
         << calib.s3() << ","
         << calib.s4() << ","

         << calib.tauX() << ","
         << calib.tauY() << ","
         << calib.reprojectionError()
         << "\n";
}
//-----------------------------------------------------------------------------
/*! Camera calibration app using opencv HighGUI and camera calibration toolbox
*/
int main(int argc, char* argv[])
{
    try
    {
        //parse arguments
        readArgs(argc, argv);

        if (helpRequired)
        {
            printHelp();
            return 0;
        }

        //---------------------------------------------------------------------
        //chessboard designer functionality
        if (designMeAChessboard)
        {
            designChessboard(chessboardSize.width, chessboardSize.height);
            return 0;
        }
        //---------------------------------------------------------------------
        cv::Scalar s = CV_RGB(255, 0, 0);

        //if a directory containing pictures was passed we use them for calibration
        if (!pictureDir.empty())
        {
            if (!Utils::dirExists(pictureDir))
                throw std::runtime_error("ERROR: pictureDir was defined but does not exist!");

            pictureDir = Utils::unifySlashes(pictureDir);
            std::cout << "Loading files from dir: " << pictureDir << std::endl;
            std::vector<std::string> namesInDir = Utils::getFileNamesInDir(pictureDir);

            std::string outputDir = Utils::unifySlashes(pictureDir + "/calibOutput");
            if (!Utils::dirExists(outputDir))
                Utils::makeDir(outputDir);

            for (int i = 0; i < namesInDir.size(); ++i)
            {
                std::cout << "Processing file: " << namesInDir[i] << std::endl;
                cv::Mat frame;
                try
                {
                    frame = cv::imread(namesInDir[i]);
                    //rotate frame if not panoramic
                    //if(frame.rows > frame.cols)
                    //    cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
                }
                catch (...)
                {
                    std::cout << "ERROR: Reading image failed!" << std::endl;
                }

                if (!frame.empty())
                {
                    if (!calibMgr)
                        calibMgr = std::make_unique<CamCalibrationManager>(chessboardSize,
                                                                           frame.size(),
                                                                           squareLength,
                                                                           namesInDir.size(),
                                                                           useReleaseObjectsMethod);

                    //extract corners and add them to manager
                    extractCorners(frame, outputDir);
                }
            }

            if (calibMgr)
            {
                int           i = 0;
                std::ofstream file;
                file.open(outputDir + "stats.csv");
                //write header
                file << "fixAspectRatio,zeroTangentDistortion,fixPrincipalPoint,calibRationalModel,calibTiltedModel,calibThinPrismModel,fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tauX,tauY,reprojError\n";

                calibAndSave(true, true, true, false, false, false, outputDir, i++, file);
                calibAndSave(true, true, false, false, false, false, outputDir, i++, file);
                calibAndSave(true, false, true, false, false, false, outputDir, i++, file);
                calibAndSave(false, true, true, false, false, false, outputDir, i++, file);
                calibAndSave(true, false, false, false, false, false, outputDir, i++, file);
                calibAndSave(false, true, false, false, false, false, outputDir, i++, file);
                calibAndSave(false, false, true, false, false, false, outputDir, i++, file);
                calibAndSave(false, false, false, false, false, false, outputDir, i++, file);

                calibAndSave(false, false, false, true, false, false, outputDir, i++, file);
                calibAndSave(false, false, false, false, true, false, outputDir, i++, file);
                calibAndSave(false, false, false, false, false, true, outputDir, i++, file);
                calibAndSave(false, false, false, true, false, true, outputDir, i++, file);
                calibAndSave(false, false, false, true, true, false, outputDir, i++, file);
                calibAndSave(false, false, false, false, true, true, outputDir, i++, file);
                calibAndSave(false, false, false, true, true, true, outputDir, i++, file);
                file.close();
                calibrated = true;
            }
        }
        else //we use opencv capture
        {
            //initialize calibration manager with default image size
            calibMgr = std::make_unique<CamCalibrationManager>(chessboardSize, defaultCamResolution, squareLength, defaultNumberOfPictures, useReleaseObjectsMethod);

            //try to instantiate video caputure tool with read arguments
            if (!initCaptureTool())
                return 0;

            cv::namedWindow(windowName);
            cv::setMouseCallback(windowName, onMouseCallback);

            cv::Mat frame;
            while (char(cv::waitKey(1)) != 'q' && cap->isOpened())
            {
                if (calibrated)
                    break;

                //get next frame
                *cap >> frame;
                cv::imwrite("C:/Users/ghm1/Development/SLProject/apps/app-CameraCalibration/test.jpg", frame);
                // rotate frame:
                // cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);

                //break if frame is empty
                if (frame.empty())
                {
                    if (videoLoop)
                    {
                        std::cout << "Video over. Starting video again..." << std::endl;
                        cap->set(cv::CAP_PROP_POS_FRAMES, 0);
                        continue;
                    }
                    else
                    {
                        std::cout << "Video over" << std::endl;
                        break;
                    }
                }

                std::vector<cv::Point2f> corners;
                bool                     foundCorners = cv::findChessboardCorners(frame, chessboardSize, corners, cv::CALIB_CB_FAST_CHECK);
                cv::drawChessboardCorners(frame, chessboardSize, cv::Mat(corners), foundCorners);
                drawStatusMessage(frame);
                //display video image
                cv::imshow(windowName, frame);
            }

            cv::destroyWindow(windowName);
        }
    }
    catch (cv::Exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown exception!" << std::endl;
    }

    return 0;
}
