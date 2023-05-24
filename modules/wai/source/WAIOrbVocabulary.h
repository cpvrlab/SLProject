//#############################################################################
//  File:      WAIOrbVocabulary.h
//  Authors:   Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef WAI_ORBVOCABULARY_H
#define WAI_ORBVOCABULARY_H
#define USE_FBOW 1

#include <string>
#include <WAIHelper.h>

#if USE_FBOW
#    include <fbow.h>
#    include <vocabulary_creator.h>
#else
#    include <orb_slam/ORBVocabulary.h>
#endif

struct WAIBowVector
{
    bool isFill = false;
    WAIBowVector(){};
    WAIBowVector(std::vector<int> wid, std::vector<float> values);
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
    WAIFeatVector(){};
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
    WAIOrbVocabulary(int layer = 2);
    ~WAIOrbVocabulary();
    void loadFromFile(std::string strVocFile);
    void create(std::vector<cv::Mat>& features, int k, int l);

#if USE_FBOW
    fbow::Vocabulary* _vocabulary = nullptr;
#else
    ORB_SLAM2::ORBVocabulary* _vocabulary = nullptr;
#endif
    void   transform(const cv::Mat& descriptors, WAIBowVector& bow, WAIFeatVector& feat);
    double score(WAIBowVector& bow1, WAIBowVector& bow2);
    size_t size();
    void   save(std::string path);
    void   setLayer(int layer) { _layer = layer; }

private:
    int _layer;
};

#endif // !WAI_ORBVOCABULARY_H
