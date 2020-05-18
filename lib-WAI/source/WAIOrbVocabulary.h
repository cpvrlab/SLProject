//#############################################################################
//  File:      WAIOrbVocabulary.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAI_ORBVOCABULARY_H
#define WAI_ORBVOCABULARY_H
#define USE_FBOW 1

#include <string>
#include <WAIHelper.h>

#if USE_FBOW
#    include <fbow.h>
#else
#    include <OrbSlam/ORBVocabulary.h>
#endif

struct WAIBowVector
{
    bool isFill = false;
#if USE_FBOW
    fbow::fBow  data;
    fbow::fBow& getWordScoreMapping() { return data; }
#else
    DBoW2::BowVector          data;
    DBoW2::BowVector&         getWordScoreMapping() { return data; }
#endif
};

struct WAIFeatVector
{
    bool isFill = false;
#if USE_FBOW
    fbow::fBow2  data;
    fbow::fBow2& getFeatMapping() { return data; }
#else
    DBoW2::FeatureVector      data;
    DBoW2::FeatureVector&     getFeatMapping() { return data; }
#endif
};

class WAI_API WAIOrbVocabulary
{
public:
    WAIOrbVocabulary();
    ~WAIOrbVocabulary();
    void loadFromFile(std::string strVocFile);

#if USE_FBOW
    fbow::Vocabulary* _vocabulary = nullptr;
#else
    ORB_SLAM2::ORBVocabulary* _vocabulary = nullptr;
#endif

    void   transform(const cv::Mat& descriptors, WAIBowVector& bow, WAIFeatVector& feat);
    double score(WAIBowVector& bow1, WAIBowVector& bow2);
    size_t size();
};

#endif // !WAI_ORBVOCABULARY_H
