/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <AverageTiming.h>
#include <BRIEFPattern.h>
#include <BRIEFextractor.h>
#include <ExtractorNode.h>

#ifdef _WINDOWS
#    include <iterator>
#endif

#include <iostream>

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE      = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD  = 19;

static void computeBriefDescriptor(const KeyPoint& kpt,
                                   const Mat&      img,
                                   const Point*    pattern,
                                   uchar*          desc)
{
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int    step   = (int)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].y) * step + cvRound(pattern[idx].x)]

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0  = GET_VALUE(0);
        t1  = GET_VALUE(1);
        val = t0 < t1;
        t0  = GET_VALUE(2);
        t1  = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

BRIEFextractor::BRIEFextractor(int   _nfeatures,
                               float _scaleFactor,
                               int   _nlevels,
                               int   _iniThFAST,
                               int   _minThFAST)
  : iniThFAST(_iniThFAST),
    minThFAST(_minThFAST),
    KPextractor("FAST-BRIEF-" + std::to_string(_nfeatures), false)
{
    nfeatures   = _nfeatures;
    scaleFactor = _scaleFactor;
    nlevels     = _nlevels;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i] = (float)(mvScaleFactor[i - 1] * scaleFactor);
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor                   = 1.0f / (float)scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    const int    npoints  = 512;
    const Point* pattern0 = (const Point*)bit_pattern_31;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int          v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2  = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

vector<cv::KeyPoint> BRIEFextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& N, const int& level)
{
    // Compute how many initial nodes
    const int nIni = (int)round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i((int)(hX * static_cast<float>(i)), 0);
        ni.UR = cv::Point2i((int)(hX * static_cast<float>(i + 1)), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[(int)(kp.pt.x / hX)]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1)
        {
            lit->bNoMore = true;
            lit++;
        }
        else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize = (int)lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end())
        {
            if (lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
        {
            bFinish = true;
        }
        else if (((int)lNodes.size() + nToExpand * 3) > N)
        {

            while (!bFinish)
            {

                prevSize = (int)lNodes.size();

                vector<pair<int, ExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = (int)vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        vector<cv::KeyPoint>& vNodeKeys   = lit->vKeys;
        cv::KeyPoint*         pKP         = &vNodeKeys[0];
        float                 maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP         = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

/**
     * 1. Splits every level of the image into evenly sized cells
     * 2. Detects corners in a 7x7 cell area
     * 3. Make sure key points are well distributed
     * 4. Compute orientation of keypoints
     * @param allKeypoints
     */
void BRIEFextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;

    for (int level = 0; level < nlevels; ++level)
    {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width  = (float)(maxBorderX - minBorderX);
        const float height = (float)(maxBorderY - minBorderY);

        const int nCols = (int)(width / W);
        const int nRows = (int)(height / W);
        const int wCell = (int)ceil(width / nCols);
        const int hCell = (int)ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = (float)(minBorderY + i * hCell);
            float       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = (float)maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = (float)(minBorderX + j * wCell);
                float       maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = (float)maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange((int)iniY, (int)maxY).colRange((int)iniX, (int)maxX),
                     vKeysCell,
                     iniThFAST,
                     true);

                if (vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange((int)iniY, (int)maxY).colRange((int)iniX, (int)maxX),
                         vKeysCell,
                         minThFAST,
                         true);
                }

                if (!vKeysCell.empty())
                {
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys,
                                      minBorderX,
                                      maxBorderX,
                                      minBorderY,
                                      maxBorderY,
                                      mnFeaturesPerLevel[level],
                                      level);

        const int scaledPatchSize = (int)(PATCH_SIZE * mvScaleFactor[level]);

        // Add border to coordinates and scale information
        const int nkps = (int)keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size   = (float)scaledPatchSize;
        }
    }
}

void BRIEFextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols / mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = (int)sqrt((float)nDesiredFeatures / (5 * imageRatio));
        const int levelRows = (int)(imageRatio * levelCols);

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

        const int W     = maxBorderX - minBorderX;
        const int H     = maxBorderY - minBorderY;
        const int cellW = (int)ceil((float)W / levelCols);
        const int cellH = (int)ceil((float)H / levelRows);

        const int nCells        = levelRows * levelCols;
        const int nfeaturesCell = (int)ceil((float)nDesiredFeatures / nCells);

        vector<vector<vector<KeyPoint>>> cellKeyPoints(levelRows, vector<vector<KeyPoint>>(levelCols));

        vector<vector<int>>  nToRetain(levelRows, vector<int>(levelCols, 0));
        vector<vector<int>>  nTotal(levelRows, vector<int>(levelCols, 0));
        vector<vector<bool>> bNoMore(levelRows, vector<bool>(levelCols, false));
        vector<int>          iniXCol(levelCols);
        vector<int>          iniYRow(levelRows);
        int                  nNoMore       = 0;
        int                  nToDistribute = 0;

        float hY = (float)(cellH + 6);

        for (int i = 0; i < levelRows; i++)
        {
            const float iniY = (float)(minBorderY + i * cellH - 3);
            iniYRow[i]       = (int)iniY;

            if (i == levelRows - 1)
            {
                hY = maxBorderY + 3 - iniY;
                if (hY <= 0)
                    continue;
            }

            float hX = (float)(cellW + 6);

            for (int j = 0; j < levelCols; j++)
            {
                float iniX;

                if (i == 0)
                {
                    iniX       = (float)(minBorderX + j * cellW - 3);
                    iniXCol[j] = (int)iniX;
                }
                else
                {
                    iniX = (float)iniXCol[j];
                }

                if (j == levelCols - 1)
                {
                    hX = maxBorderX + 3 - iniX;
                    if (hX <= 0)
                        continue;
                }

                Mat cellImage = mvImagePyramid[level].rowRange((int)iniY, (int)(iniY + hY)).colRange((int)iniX, (int)(iniX + hX));

                cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

                FAST(cellImage, cellKeyPoints[i][j], iniThFAST, true);

                if (cellKeyPoints[i][j].size() <= 3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage, cellKeyPoints[i][j], minThFAST, true);
                }

                const int nKeys = (int)cellKeyPoints[i][j].size();
                nTotal[i][j]    = nKeys;

                if (nKeys > nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j]   = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell - nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }
            }
        }

        // Retain by score

        while (nToDistribute > 0 && nNoMore < nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + (int)ceil((float)nToDistribute / (nCells - nNoMore));
            nToDistribute        = 0;

            for (int i = 0; i < levelRows; i++)
            {
                for (int j = 0; j < levelCols; j++)
                {
                    if (!bNoMore[i][j])
                    {
                        if (nTotal[i][j] > nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j]   = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell - nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures * 2);

        const int scaledPatchSize = (int)(PATCH_SIZE * mvScaleFactor[level]);

        // Retain by score and transform coordinates
        for (int i = 0; i < levelRows; i++)
        {
            for (int j = 0; j < levelCols; j++)
            {
                vector<KeyPoint>& keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
                if ((int)keysCell.size() > nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);

                for (size_t k = 0, kend = keysCell.size(); k < kend; k++)
                {
                    keysCell[k].pt.x += iniXCol[j];
                    keysCell[k].pt.y += iniYRow[i];
                    keysCell[k].octave = level;
                    keysCell[k].size   = (float)scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if ((int)keypoints.size() > nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, const vector<Point>& pattern)
{
    for (size_t i = 0; i < keypoints.size(); i++)
        computeBriefDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

void BRIEFextractor::computeKeyPointDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    descriptors.create((int)keypoints.size(), 32, CV_8U);
    computeDescriptors(image, keypoints, descriptors, pattern);
}

void BRIEFextractor::operator()(InputArray _image, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
{
    if (_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    AVERAGE_TIMING_START("ComputePyramid");
    ComputePyramid(image);
    AVERAGE_TIMING_STOP("ComputePyramid");

    vector<vector<KeyPoint>> allKeypoints;
    AVERAGE_TIMING_START("ComputeKeyPointsOctTree");
    ComputeKeyPointsOctTree(allKeypoints);
    AVERAGE_TIMING_STOP("ComputeKeyPointsOctTree");

    //ComputeKeyPointsOld(allKeypoints);

    AVERAGE_TIMING_START("BlurAndComputeDescr");
    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if (nkeypoints == 0)
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        int               tOffset         = level * 3;
        vector<KeyPoint>& keypoints       = allKeypoints[level];
        int               nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);

        computeDescriptors(workingMat, keypoints, desc, pattern);
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint    = keypoints.begin(),
                                            keypointEnd = keypoints.end();
                 keypoint != keypointEnd;
                 ++keypoint)
                keypoint->pt *= scale;
        }

        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }

    AVERAGE_TIMING_STOP("BlurAndComputeDescr");
}

void BRIEFextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size  sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size  wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat   temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if (level != 0)
        {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101 + BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101);
        }
    }

    ////save image pyriamid
    //for (int level = 0; level < nlevels; ++level) {
    //    string filename = "D:/Development/ORB_SLAM2/debug_ouput/imagePyriamid" + std::to_string(level) + ".jpg";
    //    cv::imwrite(filename, mvImagePyramid[level]);
    //}
}

} //namespace ORB_SLAM
