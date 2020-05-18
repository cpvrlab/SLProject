#include <FeatureExtractorFactory.h>
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

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(ExtractorType id, const cv::Size& videoFrameSize)
{
    switch (id)
    {
        case ExtractorType_SURF_BRIEF_500:
            return surfExtractor(500);
        case ExtractorType_SURF_BRIEF_800:
            return surfExtractor(800);
        case ExtractorType_SURF_BRIEF_1000:
            return surfExtractor(1000);
        case ExtractorType_SURF_BRIEF_1200:
            return surfExtractor(1200);
        case ExtractorType_FAST_ORBS_1000:
            return orbExtractor(1000);
        case ExtractorType_FAST_ORBS_2000:
            return orbExtractor(2000);
        case ExtractorType_FAST_ORBS_4000:
            return orbExtractor(4000);
        case ExtractorType_GLSL_1:
            return glslExtractor(videoFrameSize, 16, 16, 0.5f, 0.10f, 1.9f, 1.3f);
        case ExtractorType_GLSL:
            return glslExtractor(videoFrameSize, 16, 16, 0.5f, 0.10f, 1.9f, 1.4f);
        default:
            return surfExtractor(1000);
    }
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(std::string extractorType, const cv::Size& videoFrameSize)
{
    std::unique_ptr<KPextractor> result = nullptr;

    for (int i = 0; i < _extractorIdToNames.size(); i++)
    {
        if (_extractorIdToNames[i] == extractorType)
        {
            result = make((ExtractorType)i, videoFrameSize);
            break;
        }
    }

    return result;
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::orbExtractor(int nf)
{
    float fScaleFactor = 1.2;
    int   nLevels      = 8;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;
    return std::make_unique<ORB_SLAM2::ORBextractor>(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::surfExtractor(int th)
{
    return std::make_unique<ORB_SLAM2::SURFextractor>(th);
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::glslExtractor(const cv::Size&
                                                                      videoFrameSize,
                                                                    int   nbKeypointsBigSigma,
                                                                    int   nbKeypointsSmallSigma,
                                                                    float highThrs,
                                                                    float lowThrs,
                                                                    float bigSigma,
                                                                    float smallSigma)
{
    // int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma
    return std::make_unique<GLSLextractor>(videoFrameSize.width,
                                           videoFrameSize.height,
                                           nbKeypointsBigSigma,
                                           nbKeypointsSmallSigma,
                                           highThrs,
                                           lowThrs,
                                           bigSigma,
                                           smallSigma);
}
