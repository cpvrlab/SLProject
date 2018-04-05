//#############################################################################
//  File:      SLCVKeyframeDB.cpp
//  Author:    Raúl Mur-Artal, Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCVKeyFrameDB.h>
//-----------------------------------------------------------------------------
SLCVKeyFrameDB::SLCVKeyFrameDB(const ORBVocabulary &voc) :
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}
//-----------------------------------------------------------------------------
SLCVKeyFrameDB::~SLCVKeyFrameDB()
{
    //for (SLCVKeyFrame* kf : _keyFrames) {
    //    if (kf) 
    //        delete kf;
    //}
}
//-----------------------------------------------------------------------------
void SLCVKeyFrameDB::add(SLCVKeyFrame* pKF)
{
    //unique_lock<mutex> lock(mMutex);

    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);

    //add pointer to keyframe db
    pKF->setKeyFrameDB(this);

    //ghm1
    _keyFrames.push_back(pKF);
}
//-----------------------------------------------------------------------------
void SLCVKeyFrameDB::erase(SLCVKeyFrame* pKF)
{
    //unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
    {
        // List of keyframes that share the word
        list<SLCVKeyFrame*> &lKFs = mvInvertedFile[vit->first];

        for (list<SLCVKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
        {
            if (pKF == *lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}
//-----------------------------------------------------------------------------
void SLCVKeyFrameDB::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());

    //for (SLCVKeyFrame* kf : _keyFrames) {
    //    if (kf)
    //        delete kf;
    //}
    _keyFrames.clear();
}
//-----------------------------------------------------------------------------
vector<SLCVKeyFrame*> SLCVKeyFrameDB::DetectRelocalizationCandidates(SLCVFrame *F)
{
    list<SLCVKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        //unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end(); vit != vend; vit++)
        {
            list<SLCVKeyFrame*> &lKFs = mvInvertedFile[vit->first];

            for (list<SLCVKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                SLCVKeyFrame* pKFi = *lit;
                if (pKFi->mnRelocQuery != F->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if (lKFsSharingWords.empty())
        return vector<SLCVKeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<SLCVKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float, SLCVKeyFrame*> > lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<SLCVKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        SLCVKeyFrame* pKFi = *lit;

        if (pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<SLCVKeyFrame*>();

    list<pair<float, SLCVKeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, SLCVKeyFrame*> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        SLCVKeyFrame* pKFi = it->second;
        vector<SLCVKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        SLCVKeyFrame* pBestKF = pKFi;
        for (vector<SLCVKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            SLCVKeyFrame* pKF2 = *vit;
            if (pKF2->mnRelocQuery != F->mnId)
                continue;

            accScore += pKF2->mRelocScore;
            if (pKF2->mRelocScore>bestScore)
            {
                pBestKF = pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore>bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<SLCVKeyFrame*> spAlreadyAddedKF;
    vector<SLCVKeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, SLCVKeyFrame*> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float &si = it->first;
        if (si>minScoreToRetain)
        {
            SLCVKeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}