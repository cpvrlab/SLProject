#ifndef FEATURE_EXTRACTOR_FACTORY
#define FEATURE_EXTRACTOR_FACTORY

#include <map>
#include <string>
#include <memory>

#include <KPextractor.h>

enum ExtractorType
{
    ExtractorType_FAST_ORBS_1000  = 0,
    ExtractorType_FAST_ORBS_2000  = 1,
    ExtractorType_FAST_ORBS_3000  = 2,
    ExtractorType_FAST_ORBS_4000  = 3,
    ExtractorType_FAST_BRIEF_1000 = 4,
    ExtractorType_FAST_BRIEF_2000 = 5,
    ExtractorType_FAST_BRIEF_3000 = 6,
    ExtractorType_FAST_BRIEF_4000 = 7,
    ExtractorType_Last            = 8
};

class FeatureExtractorFactory
{
public:
    FeatureExtractorFactory();
    std::unique_ptr<ORB_SLAM2::KPextractor> make(ExtractorType id, const cv::Size& videoFrameSize, int nLevels);
    std::unique_ptr<ORB_SLAM2::KPextractor> make(std::string extractorType, const cv::Size& videoFrameSize, int nLevels);

    const std::vector<std::string>& getExtractorIdToNames() const
    {
        return _extractorIdToNames;
    }

private:
    std::unique_ptr<ORB_SLAM2::KPextractor> orbExtractor(int nf, int nLevels);
    std::unique_ptr<ORB_SLAM2::KPextractor> briefExtractor(int nf, int nLevels);
#ifndef TARGET_OS_IOS
    std::unique_ptr<ORB_SLAM2::KPextractor> glslExtractor(
      const cv::Size& videoFrameSize,
      int             nbKeypointsBigSigma,
      int             nbKeypointsSmallSigma,
      float           highThrs,
      float           lowThrs,
      float           bigSigma,
      float           smallSigma);
#endif

    std::vector<std::string> _extractorIdToNames;
};

#endif // FEATURE_EXTRACTOR_FACTORY
