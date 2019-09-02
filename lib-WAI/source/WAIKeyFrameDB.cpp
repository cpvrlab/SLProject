//#############################################################################
//  File:      WAIKeyframeDB.cpp
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

#include <WAIKeyFrameDB.h>

//-----------------------------------------------------------------------------
WAIKeyFrameDB::WAIKeyFrameDB(const ORBVocabulary& voc) : mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}
//-----------------------------------------------------------------------------
void WAIKeyFrameDB::add(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}
//-----------------------------------------------------------------------------
void WAIKeyFrameDB::erase(WAIKeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
    {
        // List of keyframes that share the word
        list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

        for (list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
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
void WAIKeyFrameDB::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}
//-----------------------------------------------------------------------------
// NOTE(jan): errorcode is set to:
// 0 - if candidates were found
// 1 - if no candidates with common words are found
// 2 - if no candidates with a high enough similarity score are found
vector<WAIKeyFrame*> WAIKeyFrameDB::DetectLoopCandidates(WAIKeyFrame* pKF, float minScore, int* errorCode)
{
    set<WAIKeyFrame*>  spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        {
            list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

            for (list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                WAIKeyFrame* pKFi = *lit;
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
    {
        *errorCode = LOOP_DETECTION_ERROR_NO_CANDIDATES_WITH_COMMON_WORDS;
        return vector<WAIKeyFrame*>();
    }

    list<pair<float, WAIKeyFrame*>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnLoopWords > maxCommonWords)
            maxCommonWords = (*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        WAIKeyFrame* pKFi = *lit;

        if (pKFi->mnLoopWords > minCommonWords)
        {
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
    {
        *errorCode = LOOP_DETECTION_ERROR_NO_SIMILAR_CANDIDATES;
        return vector<WAIKeyFrame*>();
    }

    list<pair<float, WAIKeyFrame*>> lAccScoreAndMatch;
    float                           bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (list<pair<float, WAIKeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        WAIKeyFrame*         pKFi     = it->second;
        vector<WAIKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float        bestScore = it->first;
        float        accScore  = it->first;
        WAIKeyFrame* pBestKF   = pKFi;
        for (vector<WAIKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            WAIKeyFrame* pKF2 = *vit;
            if (pKF2->mnLoopQuery == pKF->mnId && pKF2->mnLoopWords > minCommonWords)
            {
                accScore += pKF2->mLoopScore;
                if (pKF2->mLoopScore > bestScore)
                {
                    pBestKF   = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    set<WAIKeyFrame*>    spAlreadyAddedKF;
    vector<WAIKeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (list<pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        if (it->first > minScoreToRetain)
        {
            WAIKeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    *errorCode = LOOP_DETECTION_ERROR_NONE;
    return vpLoopCandidates;
}
//-----------------------------------------------------------------------------
vector<WAIKeyFrame*> WAIKeyFrameDB::DetectRelocalizationCandidates(WAIFrame* F)
{
    list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = F->mBowVec.begin(), vend = F->mBowVec.end(); vit != vend; vit++)
        {
            list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

            for (list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                WAIKeyFrame* pKFi = *lit;
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
        return vector<WAIKeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    int minCommonWords = maxCommonWords * 0.8f;

    list<pair<float, WAIKeyFrame*>> lScoreAndMatch;

    int nscores = 0;

    // Compute similarity score.
    for (list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        WAIKeyFrame* pKFi = *lit;

        if (pKFi->mnRelocWords > minCommonWords)
        {
            nscores++;
            float si          = mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;
            lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<WAIKeyFrame*>();

    list<pair<float, WAIKeyFrame*>> lAccScoreAndMatch;
    float                           bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for (list<pair<float, WAIKeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        WAIKeyFrame*         pKFi     = it->second;
        vector<WAIKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float        bestScore = it->first;
        float        accScore  = bestScore;
        WAIKeyFrame* pBestKF   = pKFi;
        for (vector<WAIKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
        {
            WAIKeyFrame* pKF2 = *vit;

            //TODO (luc)
            //Evaluate which is the best between avoid bad neighbour or compute the reloc score for all neighbours.
            if (pKF2->mnRelocQuery != F->mnId && pKF2->mnRelocWords <= minCommonWords)
                continue;

            accScore += pKF2->mRelocScore;
            if (pKF2->mRelocScore > bestScore)
            {
                pBestKF   = pKF2;
                bestScore = pKF2->mRelocScore;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float                minScoreToRetain = 0.75f * bestAccScore;
    set<WAIKeyFrame*>    spAlreadyAddedKF;
    vector<WAIKeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float& si = it->first;
        if (si > minScoreToRetain)
        {
            WAIKeyFrame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}
