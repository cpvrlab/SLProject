//#############################################################################
//  File:      WAIKeyframeDB.cpp
//  Authors:   Raúl Mur-Artal, Michael Goettlicher
//  Date:      October 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
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
#include <string>
#include <sstream>
//-----------------------------------------------------------------------------
WAIKeyFrameDB::WAIKeyFrameDB(WAIOrbVocabulary* voc) : mpVoc(voc)
{
    mvInvertedFile.resize(mpVoc->size());
}

//-----------------------------------------------------------------------------
void WAIKeyFrameDB::add(WAIKeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);
    if (pKF->mBowVec.data.empty())
    {
        std::cout << "kf data empty" << std::endl;
        return;
    }
    for (auto vit = pKF->mBowVec.getWordScoreMapping().begin(), vend = pKF->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
    {
        mvInvertedFile[vit->first].push_back(pKF);
    }
}
//-----------------------------------------------------------------------------
void WAIKeyFrameDB::erase(WAIKeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (auto vit = pKF->mBowVec.getWordScoreMapping().begin(), vend = pKF->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
    {
        // List of keyframes that share the word
        std::list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

        for (std::list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
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
std::vector<WAIKeyFrame*> WAIKeyFrameDB::DetectLoopCandidates(WAIKeyFrame* pKF, float minCommonWordFactor, float minScore, int* errorCode)
{
    std::set<WAIKeyFrame*>  spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    std::list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (auto vit = pKF->mBowVec.getWordScoreMapping().begin(), vend = pKF->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
        {
            std::list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

            for (std::list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
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
        return std::vector<WAIKeyFrame*>();
    }

    std::list<std::pair<float, WAIKeyFrame*>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnLoopWords > maxCommonWords)
            maxCommonWords = (*lit)->mnLoopWords;
    }

    int minCommonWords = (int)(maxCommonWords * minCommonWordFactor);

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        WAIKeyFrame* pKFi = *lit;

        if (pKFi->mnLoopWords > minCommonWords)
        {
            float si = (float)mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(std::make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
    {
        *errorCode = LOOP_DETECTION_ERROR_NO_SIMILAR_CANDIDATES;
        return std::vector<WAIKeyFrame*>();
    }

    std::list<std::pair<float, WAIKeyFrame*>> lAccScoreAndMatch;
    float                                     bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        WAIKeyFrame*              pKFi     = it->second;
        std::vector<WAIKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float        bestScore = it->first;
        float        accScore  = it->first;
        WAIKeyFrame* pBestKF   = pKFi;
        for (std::vector<WAIKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
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

        lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    std::set<WAIKeyFrame*>    spAlreadyAddedKF;
    std::vector<WAIKeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
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

std::vector<WAIKeyFrame*> WAIKeyFrameDB::DetectRelocalizationCandidates(WAIFrame* F, cv::Mat extrinsicGuess)
{
    std::list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (auto vit = F->mBowVec.getWordScoreMapping().begin(), vend = F->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
        {
            if (vit->first > mvInvertedFile.size())
            {
                std::stringstream ss;
                ss << "WAIKeyFrameDB::DetectRelocalizationCandidates: word index bigger than inverted file. word: " << vit->first << " val: " << vit->second;
                throw std::runtime_error(ss.str());
            }

            std::list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

            for (std::list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                WAIKeyFrame* pKFi = *lit;
                if (pKFi->isBad())
                    continue;

                if (pKFi->mnRelocQuery != F->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mRelocScore  = 0.f;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if (lKFsSharingWords.empty())
        return std::vector<WAIKeyFrame*>();

    std::vector<WAIKeyFrame*> kfs;
    for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        kfs.push_back(*lit);
    }
    return kfs;
}

std::vector<WAIKeyFrame*> WAIKeyFrameDB::DetectRelocalizationCandidates(WAIFrame* F, float minCommonWordFactor, bool applyMinAccScoreFilter)
{
    std::list<WAIKeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (auto vit = F->mBowVec.getWordScoreMapping().begin(), vend = F->mBowVec.getWordScoreMapping().end(); vit != vend; vit++)
        {
            if (vit->first > mvInvertedFile.size())
            {
                std::stringstream ss;
                ss << "WAIKeyFrameDB::DetectRelocalizationCandidates: word index bigger than inverted file. word: " << vit->first << " val: " << vit->second;
                throw std::runtime_error(ss.str());
            }

            std::list<WAIKeyFrame*>& lKFs = mvInvertedFile[vit->first];

            for (std::list<WAIKeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
            {
                WAIKeyFrame* pKFi = *lit;
                if (pKFi->isBad())
                    continue;

                if (pKFi->mnRelocQuery != F->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mRelocScore  = 0.f;
                    pKFi->mnRelocQuery = F->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++;
            }
        }
    }
    if (lKFsSharingWords.empty())
        return std::vector<WAIKeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
    {
        if ((*lit)->mnRelocWords > maxCommonWords)
            maxCommonWords = (*lit)->mnRelocWords;
    }

    int minCommonWords = (int)(maxCommonWords * minCommonWordFactor);

    if (!applyMinAccScoreFilter)
    {
        std::vector<WAIKeyFrame*> vpRelocCandidates;

        // Compute similarity score.
        for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
        {
            WAIKeyFrame* pKFi = *lit;

            if (pKFi->mnRelocWords > minCommonWords)
            {
                vpRelocCandidates.push_back(pKFi);
            }
        }

        return vpRelocCandidates;
    }
    else
    {
        //apply minimum accumulated score filter:
        /*We group those keyframes that are connected in the covisibility graph and caluculate an accumulated score.
            We return all keyframe matches whose scores are higher than the 75 % of the best score.*/
        std::list<std::pair<float, WAIKeyFrame*>> lScoreAndMatch;
        int                                       nscores = 0;

        // Compute similarity score.
        for (std::list<WAIKeyFrame*>::iterator lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end(); lit != lend; lit++)
        {
            WAIKeyFrame* pKFi = *lit;

            if (pKFi->mnRelocWords > minCommonWords)
            {
                nscores++;
                float si = (float)mpVoc->score(F->mBowVec, pKFi->mBowVec);
                //std::cout << "si: " << si << std::endl;
                pKFi->mRelocScore = si;
                lScoreAndMatch.push_back(std::make_pair(si, pKFi));
            }
        }

        if (lScoreAndMatch.empty())
            return std::vector<WAIKeyFrame*>();

        std::list<std::pair<float, WAIKeyFrame*>> lAccScoreAndMatch;
        float                                     bestAccScore = 0;

        // Lets now accumulate score by covisibility
        for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
        {
            WAIKeyFrame*              pKFi     = it->second;
            std::vector<WAIKeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float        bestScore = it->first;
            float        accScore  = bestScore;
            WAIKeyFrame* pBestKF   = pKFi;
            //std::cout << "vpNeighs: " << vpNeighs.size() << std::endl;
            for (std::vector<WAIKeyFrame*>::iterator vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++)
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
            lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
            if (accScore > bestAccScore)
                bestAccScore = accScore;
        }

        //std::cout << "lAccScoreAndMatch: " << lAccScoreAndMatch.size() << std::endl;

        // Return all those keyframes with a score higher than 0.75*bestScore
        // This ensures that all the neighbours are also
        float minScoreToRetain = 0.75f * bestAccScore;
        //std::cout << "minScoreToRetain: " << minScoreToRetain << std::endl;
        std::set<WAIKeyFrame*>    spAlreadyAddedKF;
        std::vector<WAIKeyFrame*> vpRelocCandidates;
        vpRelocCandidates.reserve(lAccScoreAndMatch.size());
        for (std::list<std::pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
        {
            const float& si = it->first;
            //std::cout << "final si: " << si << std::endl;

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
    /*

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
        std::cout << "vpNeighs: " << vpNeighs.size() << std::endl;
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

    std::cout << "lAccScoreAndMatch: " << lAccScoreAndMatch.size() << std::endl;

    // Return all those keyframes with a score higher than 0.75*bestScore
    // This ensures that all the neighbours are also
    float minScoreToRetain = 0.75f * bestAccScore;
    std::cout << "minScoreToRetain: " << minScoreToRetain << std::endl;
    set<WAIKeyFrame*>    spAlreadyAddedKF;
    vector<WAIKeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for (list<pair<float, WAIKeyFrame*>>::iterator it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end(); it != itend; it++)
    {
        const float& si = it->first;
        std::cout << "final si: " << si << std::endl;

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
    */
}
