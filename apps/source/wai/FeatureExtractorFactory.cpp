#include "FeatureExtractorFactory.h"
#include <ORBextractor.h>
#include <SURFextractor.h>
#include <GLSLextractor.h>

using namespace ORB_SLAM2;

FeatureExtractorFactory::FeatureExtractorFactory()
{
    _extractorIdToNames.resize(ExtractorType_Last);
    _extractorIdToNames[ExtractorType_SURF_BRIEF_500]  = "SURF-BRIEF-500";
    _extractorIdToNames[ExtractorType_SURF_BRIEF_800]  = "SURF-BRIEF-800";
    _extractorIdToNames[ExtractorType_SURF_BRIEF_1000] = "SURF-BRIEF-1000";
    _extractorIdToNames[ExtractorType_SURF_BRIEF_1200] = "SURF-BRIEF-1200";
    _extractorIdToNames[ExtractorType_FAST_ORBS_1000]  = "FAST-ORBS-1000";
    _extractorIdToNames[ExtractorType_FAST_ORBS_2000]  = "FAST-ORBS-2000";
    _extractorIdToNames[ExtractorType_FAST_ORBS_4000]  = "FAST-ORBS-4000";
    _extractorIdToNames[ExtractorType_GLSL_1]          = "GLSL-1";
    _extractorIdToNames[ExtractorType_GLSL]            = "GLSL";
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(ExtractorType id, cv::Size videoFrameSize)
{
    switch (id)
    {
        case ExtractorType_SURF_BRIEF_500:
            return std::move(surfExtractor(500));
        case ExtractorType_SURF_BRIEF_800:
            return std::move(surfExtractor(800));
        case ExtractorType_SURF_BRIEF_1000:
            return std::move(surfExtractor(1000));
        case ExtractorType_SURF_BRIEF_1200:
            return std::move(surfExtractor(1200));
        case ExtractorType_FAST_ORBS_1000:
            return std::move(orbExtractor(1000));
        case ExtractorType_FAST_ORBS_2000:
            return std::move(orbExtractor(2000));
        case ExtractorType_FAST_ORBS_4000:
            return std::move(orbExtractor(4000));
        case ExtractorType_GLSL_1:
            return std::move(glslExtractor(videoFrameSize, 16, 16, 0.5, 0.25, 1.9, 1.4));
        case ExtractorType_GLSL:
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
