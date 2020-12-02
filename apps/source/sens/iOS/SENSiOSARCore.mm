#include "SENSiOSARCore.h"

SENSiOSARCore::SENSiOSARCore()
{
    _arcoreDelegate = [[SENSiOSARCoreDelegate alloc] init];
    //check availablity
    _available = [_arcoreDelegate isAvailable];
    //set update callback
    //[_arcoreDelegate setUpdateCB:std::bind(&SENSiOSARCore::onUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7)];
    //[_arcoreDelegate setUpdateBgrCB:std::bind(&SENSiOSARCore::onUpdateBGR, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)];
}

bool SENSiOSARCore::init(int targetWidth, int targetHeight, int manipWidth, int manipHeight, bool convertManipToGray)
{
    if(!_available)
        return false;
    
    configure(targetWidth, targetHeight, manipWidth, manipHeight, convertManipToGray);
    
    bool success = [_arcoreDelegate start];
    _pause = false;
    
    return success;
}

bool SENSiOSARCore::isReady()
{
    return false;
}

bool SENSiOSARCore::resume()
{
    return false;
}

void SENSiOSARCore::reset()
{
    
}

void SENSiOSARCore::pause()
{
    //todo:
    _pause = true;
}

SENSFramePtr SENSiOSARCore::latestFrame()
{
    //retrieve the latest frame from arkit delegate
    cv::Mat pose;
    cv::Mat intrinsic;
    cv::Mat imgBGR;
    bool isTracking;
    [_arcoreDelegate latestFrame:&pose withImg:&imgBGR AndIntrinsic:&intrinsic IsTracking:&isTracking];
    
    SENSFramePtr latestFrame;
    if(!imgBGR.empty())
        latestFrame = processNewFrame(SENSClock::now(), imgBGR, intrinsic, pose, isTracking);
    return latestFrame;
}

void SENSiOSARCore::setDisplaySize(int w, int h)
{
    
}

void SENSiOSARCore::onUpdateBGR(simd_float4x4* camPose, cv::Mat imgBGR, simd_float3x3* camMat3x3)
{
    cv::Mat intrinsics;
    bool    intrinsicsChanged = false;
    if (camMat3x3)
    {
        intrinsicsChanged = true;
        intrinsics        = cv::Mat_<double>(3, 3);
        for (int i = 0; i < 3; ++i)
        {
            simd_float3 col             = camMat3x3->columns[i];
            intrinsics.at<double>(0, i) = (double)col[0];
            intrinsics.at<double>(1, i) = (double)col[1];
            intrinsics.at<double>(2, i) = (double)col[2];
        }
        
        //std::cout << intrinsics << std::endl;
    }

    
    cv::Mat pose;
    if(camPose)
    {
        pose = cv::Mat_<float>(4, 4);
        for (int i = 0; i < 4; ++i)
        {
            simd_float4 col = camPose->columns[i];
            pose.at<float>(0, i) = (float)col[0];
            pose.at<float>(1, i) = (float)col[1];
            pose.at<float>(2, i) = (float)col[2];
            pose.at<float>(3, i) = (float)col[3];
        }
        
        //std::cout << pose << std::endl;
    }
    
    setFrame( std::make_shared<SENSFrameBase>(SENSClock::now(), imgBGR, intrinsics, pose, true));
}

void SENSiOSARCore::onUpdate(simd_float4x4* camPose, uint8_t* yPlane, uint8_t* uvPlane, int imgWidth, int imgHeight, simd_float3x3* camMat3x3, bool isTracking)
{
    cv::Mat intrinsics;
    bool    intrinsicsChanged = false;
    if (camMat3x3)
    {
        intrinsicsChanged = true;
        intrinsics        = cv::Mat_<double>(3, 3);
        for (int i = 0; i < 3; ++i)
        {
            simd_float3 col             = camMat3x3->columns[i];
            intrinsics.at<double>(0, i) = (double)col[0];
            intrinsics.at<double>(1, i) = (double)col[1];
            intrinsics.at<double>(2, i) = (double)col[2];
        }
        //std::cout << intrinsics << std::endl;
    }

    
    cv::Mat pose;
    if(camPose)
    {
        pose = cv::Mat_<float>(4, 4);
        for (int i = 0; i < 4; ++i)
        {
            simd_float4 col = camPose->columns[i];
            pose.at<float>(0, i) = (float)col[0];
            pose.at<float>(1, i) = (float)col[1];
            pose.at<float>(2, i) = (float)col[2];
            pose.at<float>(3, i) = (float)col[3];
        }
        //std::cout << pose << std::endl;
    }
    
    //ATTENTION:
    //-in release mode it is much faster to directly convert YUV to BGR
    //-in debug mode it is much too slow to directly convert YUV to BGR (it does not work), so we convert in an extra worker thread
#ifdef _DEBUG
    cv::Mat yuvImg((int)imgHeight + ((int)imgHeight / 2), (int)imgWidth, CV_8UC1);
    // copy interleaved y data
    size_t yLen = imgWidth * imgHeight;
    memcpy(yuvImg.data, yPlane, yLen);
    // copy interleaved uv data
    size_t uvLen = yLen / 2;
    memcpy(yuvImg.data + yLen, uvPlane, uvLen);
    //Utils::log("SENSiOSARCore", "memcpy: %fms", t.elapsedTimeInMilliSec());
    
    if(!_imgConverter)
        _imgConverter = std::make_unique<ImageConverter>(std::bind(&SENSiOSARCore::setFrame, this, std::placeholders::_1));
    //we transfer the yuv image as bgr, it will be converter in the ImageConverter
    _imgConverter->setFrame(std::make_unique<SENSFrameBase>(SENSClock::now(), yuvImg, intrinsics, pose, isTracking));
#else
    //HighResTimer t;
    cv::Mat yuvImg((int)imgHeight + ((int)imgHeight / 2), (int)imgWidth, CV_8UC1, yPlane);
    
    //we dont need to copy, cvrColor calculates into target image
    /*
    // copy interleaved y data
    size_t yLen = imgWidth * imgHeight;
    memcpy(yuvImg.data, yPlane, yLen);
    // copy interleaved uv data
    size_t uvLen = yLen / 2;
    memcpy(yuvImg.data + yLen, uvPlane, uvLen);
    //Utils::log("SENSiOSARCore", "memcpy: %fms", t.elapsedTimeInMilliSec());
    */
    //t.start();
    cv::Mat bgrImg;
    cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR_NV12, 3);
    //Utils::log("SENSiOSARCore", "convertion: %fms", t.elapsedTimeInMilliSec());
    
    setFrame( std::make_shared<SENSFrameBase>(SENSClock::now(), bgrImg, intrinsics, pose, isTracking));
#endif
    

}
