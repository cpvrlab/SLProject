//#############################################################################
//  File:      SLCVKeyframeDB.cpp
//  Author:    Raúl Mur-Artal, Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdafx.h>
#include <SLCVKeyFrameDB.h>

//-----------------------------------------------------------------------------
SLCVKeyFrameDB::SLCVKeyFrameDB(const ORBVocabulary &voc) :
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}
//-----------------------------------------------------------------------------
void SLCVKeyFrameDB::add(SLCVKeyFrame* pKF)
{
    //unique_lock<mutex> lock(mMutex);

    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
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
}
//-----------------------------------------------------------------------------
vector<SLCVKeyFrame*> SLCVKeyFrameDB::DetectLoopCandidates(SLCVKeyFrame* pKF, float minScore)
{
    set<SLCVKeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<SLCVKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        //unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<SLCVKeyFrame*> &lKFs = mvInvertedFile[vit->first];

            for (list<SLCVKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                SLCVKeyFrame* pKFi = *lit;
                if (pKFi->mnLoopQuery != pKF->mnId)
                {
                    pKFi->mnLoopWords = 0;
                    if (!spConnectedKeyFrames.count(pKFi))
                    {
                        pKFi->mnLoopQuery = pKF->mnId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<SLCVKeyFrame*>();

    list<pair<float, SLCVKeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<SLCVKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords = (*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f;

    int nscores = 0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (list<SLCVKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        SLCVKeyFrame* pKFi = *lit;

        if (pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<SLCVKeyFrame*>();

    list<pair<float, SLCVKeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (list<pair<float, SLCVKeyFrame*> >::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        SLCVKeyFrame* pKFi = it->second;
        vector<SLCVKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        SLCVKeyFrame* pBestKF = pKFi;
        for (vector<SLCVKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            SLCVKeyFrame* pKF2 = *vit;
            if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore += pKF2->mLoopScore;
                if (pKF2->mLoopScore>bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore>bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<SLCVKeyFrame*> spAlreadyAddedKF;
    vector<SLCVKeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (list<pair<float, SLCVKeyFrame*> >::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        if (it->first>minScoreToRetain)
        {
            SLCVKeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
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