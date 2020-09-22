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
    catch (std::exception& e)
    {
        std::string err = "WAIOrbVocabulary::loadFromFile: failed to load vocabulary " + strVocFile;
        throw std::runtime_error(err);
    }
#else
    bool bVocLoad = _vocabulary->loadFromBinaryFile(strVocFile);

    if (!bVocLoad)
    {
        std::string err = "WAIOrbVocabulary::loadFromFile: failed to load vocabulary " + strVocFile;
        Utils::log("WAI", err.c_str());
        throw std::runtime_error(err);
    }
#endif
}

WAIBowVector::WAIBowVector(std::vector<int> wid, std::vector<float> values)
{
    for (int i = 0; i < wid.size(); i++)
    {
#if USE_FBOW
        fbow::_float v;
        v.var = values[i];
        data.insert(std::pair<uint32_t, fbow::_float>(wid[i], v));
#else
        data.insert(std::pair<DBoW2::WordId, DBoW2::WordValue>(wid[i], values[i]));
#endif
    }
}

void WAIOrbVocabulary::transform(const cv::Mat& descriptors, WAIBowVector& bow, WAIFeatVector& feat)
{
    bow.isFill  = true;
    feat.isFill = true;

    if (descriptors.rows == 0)
        return;

#if USE_FBOW
    _vocabulary->transform(descriptors, 2, bow.data, feat.data);
#else
    vector<cv::Mat> vCurrentDesc = ORB_SLAM2::Converter::toDescriptorVector(descriptors);
    _vocabulary->transform(vCurrentDesc, bow.data, feat.data, _vocabulary->getDepthLevels() - 2);
#endif
}

double WAIOrbVocabulary::score(WAIBowVector& bow1, WAIBowVector& bow2)
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

void WAIOrbVocabulary::create(std::vector<cv::Mat> features)
{
#if USE_FBOW
    fbow::VocabularyCreator vc;
    fbow::VocabularyCreator::Params p;
    p.k        = 10;
    p.L        = 2;
    p.nthreads = 1;
    p.maxIters = 11;
    p.verbose  = true;

    _vocabulary = new fbow::Vocabulary();

    cout << "Creating a " << p.k << "^" << p.L << " vocabulary..." << endl;
    vc.create(*_vocabulary, features, "orb2000", p);
    cout << "... done!" << endl;
#else
    const int           k      = 10;
    const int           L      = 2;
    const WeightingType weight = TF_IDF;
    const ScoringType   score  = L1_NORM;

    std::vector<std::vector<cv::Mat>> feats;
    feat.resize(features.size());

    cout << "Creating a " << p.k << "^" << p.L << " vocabulary..." << endl;
    for (int i = 0; i < features.size(); i++)
    {
        feat[i].resize(features[i].rows());
        for (int j = 0; j < features[i].rows(); j++)
            feat[i].push_back(features[i].row(i));
    }

    _vocabulary = new voc(k, L, weight, score);
    _vocabulary->create(feat);

    cout << "... done!" << endl;
#endif
}

void WAIOrbVocabulary::save(std::string path)
{
#if USE_FBOW
    _vocabulary->saveToFile(path);
#else
    _vocabulary->saveToBinaryFile(path);
#endif
}
