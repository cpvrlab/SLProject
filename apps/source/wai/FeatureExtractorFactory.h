#ifndef FEATURE_EXTRACTOR_FACTORY
#define FEATURE_EXTRACTOR_FACTORY

#include <map>
#include <string>
#include <memory>

#include <KPextractor.h>

enum ExtractorType
{
    ExtractorType_SURF_BRIEF_500  = 0,
    ExtractorType_SURF_BRIEF_800  = 1,
    ExtractorType_SURF_BRIEF_1000 = 2,
    ExtractorType_SURF_BRIEF_1200 = 3,
    ExtractorType_FAST_ORBS_1000  = 4,
    ExtractorType_FAST_ORBS_2000  = 5,
    ExtractorType_FAST_ORBS_4000  = 6,
    ExtractorType_GLSL_1          = 7,
    ExtractorType_GLSL            = 8,
    ExtractorType_Last            = 9
};

class FeatureExtractorFactory
{
public:
    FeatureExtractorFactory();
    std::unique_ptr<ORB_SLAM2::KPextractor> make(int id, cv::Size videoFrameSize);

    const std::vector<std::string>& getExtractorIdToNames() const
    {
        return _extractorIdToNames;
    }

private:
    std::unique_ptr<ORB_SLAM2::KPextractor> orbExtractor(int nf);
    std::unique_ptr<ORB_SLAM2::KPextractor> surfExtractor(int th);
    std::unique_ptr<ORB_SLAM2::KPextractor> glslExtractor(
      const cv::Size& videoFrameSize,
      int             nbKeypointsBigSigma,
      int             nbKeypointsSmallSigma,
      float           highThrs,
      float           lowThrs,
      float           bigSigma,
      float           smallSigma);

    std::vector<std::string> _extractorIdToNames;
};

#endif //FEATURE_EXTRACTOR_FACTORY
