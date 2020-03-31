#include <CVTrackedWAI.h>

#include <SL.h>

CVTrackedWAI::CVTrackedWAI(std::string vocabularyFile)
{
    _voc = new ORB_SLAM2::ORBVocabulary();
    if (!_voc->loadFromBinaryFile(vocabularyFile))
    {
        SL_LOG("Could not load vocabulary file!");
    }
}

bool CVTrackedWAI::track(CVMat imageGray, CVMat imageRgb, CVCalibration* calib)
{
    bool result = false;

    float startMS = _timer.elapsedTimeInMilliSec();

    if (!_mode)
    {
        if (!_voc)
            return false;

        int   nf           = 2000;
        float fScaleFactor = 1.2;
        int   nLevels      = 8;
        int   fIniThFAST   = 20;
        int   fMinThFAST   = 7;
        _trackingExtractor = new ORB_SLAM2::ORBextractor(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        _mode = new WAISlam(calib->cameraMat(),
                            calib->distortion(),
                            _voc,
                            _trackingExtractor,
                            nullptr);
    }

    if (_mode->update(imageGray))
    {
        cv::Mat pose = _mode->getPose();

        cv::Mat rot         = cv::Mat::eye(4, 4, CV_32F);
        rot.at<float>(1, 1) = -1.0f;
        rot.at<float>(2, 2) = -1.0f;

        pose = rot * pose;

        pose.copyTo(_objectViewMat);

        result = true;
    }

    if (_drawDetection)
    {
        _mode->drawInfo(imageRgb, true, true, true);
    }

    // TODO(dgj1): at the moment we cant differentiate between these two
    // as they are both done in the same call to WAI
    CVTracked::detectTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);
    CVTracked::poseTimesMS.set(_timer.elapsedTimeInMilliSec() - startMS);

    return result;
}
