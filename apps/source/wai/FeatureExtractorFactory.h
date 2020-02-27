#ifndef FEATURE_EXTRACTOR_FACTORY
#define FEATURE_EXTRACTOR_FACTORY

#include <map>
#include <string>
#include <memory>

#include <KPextractor.h>

enum FeatureType
{
    SURF_BRIEF_500,
    SURF_BRIEF_800,
    SURF_BRIEF_1000,
    SURF_BRIEF_1200,
    FAST_ORBS_1000,
    FAST_ORBS_2000,
    FAST_ORBS_4000,
    GLSL_1,
    GLSL
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
