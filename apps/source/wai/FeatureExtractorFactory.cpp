#include "FeatureExtractorFactory.h"
#include <ORBextractor.h>
#include <SURFextractor.h>
#include <GLSLextractor.h>

using namespace ORB_SLAM2;

FeatureExtractorFactory::FeatureExtractorFactory()
{
    _extractorIdToNames.push_back("SURF-BRIEF-500");
    _extractorIdToNames.push_back("SURF-BRIEF-800");
    _extractorIdToNames.push_back("SURF-BRIEF-1000");
    _extractorIdToNames.push_back("SURF-BRIEF-1200");
    _extractorIdToNames.push_back("FAST-ORBS-1000");
    _extractorIdToNames.push_back("FAST-ORBS-2000");
    _extractorIdToNames.push_back("FAST-ORBS-4000");
    _extractorIdToNames.push_back("GLSL-1");
    _extractorIdToNames.push_back("GLSL");
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(int id, cv::Size videoFrameSize)
{
    switch (id)
    {
        case 0:
            return std::move(surfExtractor(500));
        case 1:
            return std::move(surfExtractor(800));
        case 2:
            return std::move(surfExtractor(1000));
        case 3:
            return std::move(surfExtractor(1200));
        case 4:
            return std::move(orbExtractor(1000));
        case 5:
            return std::move(orbExtractor(2000));
        case 6:
            return std::move(orbExtractor(4000));
        case 7:
            return std::move(glslExtractor(videoFrameSize, 16, 16, 0.5, 0.25, 1.9, 1.4));
        case 8:
            return std::move(glslExtractor(videoFrameSize, 16, 16, 0.5, 0.25, 1.8, 1.2));
        default:
            return std::move(surfExtractor(1000));
    }
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::orbExtractor(int nf)
{
    float fScaleFactor = 1.2;
    int   nLevels      = 8;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;
    return std::move(
      std::make_unique<ORB_SLAM2::ORBextractor>(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST));
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::surfExtractor(int th)
{
    return std::move(
      std::make_unique<ORB_SLAM2::SURFextractor>(th));
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::glslExtractor(const cv::Size& videoFrameSize, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma)
{
    // int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma
    return std::move(
      std::make_unique<GLSLextractor>(videoFrameSize.width, videoFrameSize.height, nbKeypointsBigSigma, nbKeypointsSmallSigma, highThrs, lowThrs, bigSigma, smallSigma));
}
