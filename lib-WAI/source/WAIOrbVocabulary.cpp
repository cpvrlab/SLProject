//#############################################################################
//  File:      WAIOrbVocabulary.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <Converter.h>
#include <WAIOrbVocabulary.h>
#include <Utils.h>

WAIOrbVocabulary::WAIOrbVocabulary()
{
#if USE_FBOW
    _vocabulary = new fbow::Vocabulary();
#else
    _vocabulary = new ORB_SLAM2::ORBVocabulary();
#endif
}

WAIOrbVocabulary::~WAIOrbVocabulary()
{
    if (_vocabulary)
    {
#if USE_FBOW
        _vocabulary->clear();
#endif
        delete _vocabulary;
    }

    _vocabulary = nullptr;
}

//-----------------------------------------------------------------------------
void WAIOrbVocabulary::loadFromFile(std::string strVocFile)
{
    if (!_vocabulary)
    {
#if USE_FBOW
        _vocabulary = new fbow::Vocabulary();
#else
        _vocabulary = new ORB_SLAM2::ORBVocabulary();
#endif
    }

#if USE_FBOW
    try
    {
        _vocabulary->readFromFile(strVocFile);
    }
    catch(std::exception& e)
    {
        std::string err = "WAIOrbVocabulary::loadFromFile: failed to load vocabulary " + strVocFile;
        throw std::runtime_error(err);
    }
#else
    bool bVocLoad = _vocabulary->loadFromBinaryFile(strVocFile);

    if (!bVocLoad)
    {
        std::string err = "WAIOrbVocabulary::loadFromFile: failed to load vocabulary " + strVocFile;
        throw std::runtime_error(err);
    }
#endif
}

void WAIOrbVocabulary::transform(const cv::Mat &descriptors, WAIBowVector &bow, WAIFeatVector &feat)
{
    bow.isFill = true;
    feat.isFill = true;

    if(descriptors.rows == 0)
        return;

#if USE_FBOW
    _vocabulary->transform(descriptors, 1, bow.data, feat.data);
#else
    vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(descriptors);
    _vocabulary->transform(vCurrentDesc, bow.data, feat.data, _vocabulary->getDepthLevels() - 2);
#endif
}

double WAIOrbVocabulary::score(WAIBowVector &bow1, WAIBowVector &bow2)
{
#if USE_FBOW
    return fbow::fBow::score(bow1.data, bow2.data);
#else
    return _vocabulary->score(bow1.data, bow2.data);
#endif
}

size_t WAIOrbVocabulary::size()
{
#if USE_FBOW
    return _vocabulary->size() * _vocabulary->getK();
#else
    return _vocabulary->size();
#endif
}



