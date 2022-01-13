#include <FeatureExtractorFactory.h>
#include <ORBextractor.h>
#include <BRIEFextractor.h>
#include <GLSLextractor.h>

using namespace ORB_SLAM2;

FeatureExtractorFactory::FeatureExtractorFactory()
{
    _extractorIdToNames.resize(ExtractorType_Last);
    _extractorIdToNames[ExtractorType_FAST_ORBS_1000]  = "FAST-ORBS-1000";
    _extractorIdToNames[ExtractorType_FAST_ORBS_2000]  = "FAST-ORBS-2000";
    _extractorIdToNames[ExtractorType_FAST_ORBS_3000]  = "FAST-ORBS-3000";
    _extractorIdToNames[ExtractorType_FAST_ORBS_4000]  = "FAST-ORBS-4000";
    _extractorIdToNames[ExtractorType_FAST_BRIEF_1000] = "FAST-BRIEF-1000";
    _extractorIdToNames[ExtractorType_FAST_BRIEF_2000] = "FAST-BRIEF-2000";
    _extractorIdToNames[ExtractorType_FAST_BRIEF_3000] = "FAST-BRIEF-3000";
    _extractorIdToNames[ExtractorType_FAST_BRIEF_4000] = "FAST-BRIEF-4000";
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(ExtractorType id, const cv::Size& videoFrameSize, int nLevels)
{
    switch (id)
    {
        case ExtractorType_FAST_ORBS_1000:
            return orbExtractor(1000, nLevels);
        case ExtractorType_FAST_ORBS_2000:
            return orbExtractor(2000, nLevels);
        case ExtractorType_FAST_ORBS_3000:
            return orbExtractor(3000, nLevels);
        case ExtractorType_FAST_ORBS_4000:
            return orbExtractor(4000, nLevels);
        case ExtractorType_FAST_BRIEF_1000:
            return briefExtractor(1000, nLevels);
        case ExtractorType_FAST_BRIEF_2000:
            return briefExtractor(2000, nLevels);
        case ExtractorType_FAST_BRIEF_3000:
            return briefExtractor(3000, nLevels);
        case ExtractorType_FAST_BRIEF_4000:
            return briefExtractor(4000, nLevels);
            /*
            #ifndef TARGET_OS_IOS
                    case ExtractorType_GLSL_1:
                        return glslExtractor(videoFrameSize, 16, 16, 0.5f, 0.10f, 1.9f, 1.3f);
                    case ExtractorType_GLSL:
                        return glslExtractor(videoFrameSize, 16, 16, 0.5f, 0.10f, 1.9f, 1.4f);
            #endif
            */
        default:
            return orbExtractor(1000, nLevels);
    }
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::make(std::string extractorType, const cv::Size& videoFrameSize, int nLevels)
{
    std::unique_ptr<KPextractor> result = nullptr;

    for (int i = 0; i < _extractorIdToNames.size(); i++)
    {
        if (_extractorIdToNames[i] == extractorType)
        {
            result = make((ExtractorType)i, videoFrameSize, nLevels);
            break;
        }
    }

    return result;
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::orbExtractor(int nf, int nLevels)
{
    float fScaleFactor = 1.2f;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;
    return std::make_unique<ORB_SLAM2::ORBextractor>(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
}

std::unique_ptr<KPextractor> FeatureExtractorFactory::briefExtractor(int nf, int nLevels)
{
    float fScaleFactor = 1.2f;
    int   fIniThFAST   = 20;
    int   fMinThFAST   = 7;
    return std::make_unique<ORB_SLAM2::BRIEFextractor>(nf, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
}

#ifndef TARGET_OS_IOS
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
#endif
