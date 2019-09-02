#include "ExtractKeypoints.h"
#include "convert.h"
#include <cstdlib>

typedef struct QuadTreeNode
{
    QuadTreeNode(){bNoMore = false;}
    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<QuadTreeNode>::iterator lit;
    bool bNoMore;
}QuadTreeNode;


static void DivideNode(QuadTreeNode * p, QuadTreeNode& n1, QuadTreeNode& n2, QuadTreeNode& n3, QuadTreeNode& n4)
{
    const int halfX = ceil(static_cast<float>(p->UR.x - p->UL.x) / 2);
    const int halfY = ceil(static_cast<float>(p->BR.y - p->UL.y) / 2);

    //Define boundaries of childs
    n1.UL = p->UL;
    n1.UR = cv::Point2i(p->UL.x + halfX, p->UL.y);
    n1.BL = cv::Point2i(p->UL.x, p->UL.y + halfY);
    n1.BR = cv::Point2i(p->UL.x + halfX, p->UL.y + halfY);
    n1.vKeys.reserve(p->vKeys.size());
    n1.bNoMore = false;

    n2.UL = n1.UR;
    n2.UR = p->UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(p->UR.x, p->UL.y + halfY);
    n2.vKeys.reserve(p->vKeys.size());
    n2.bNoMore = false;

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = p->BL;
    n3.BR = cv::Point2i(n1.BR.x, p->BL.y);
    n3.vKeys.reserve(p->vKeys.size());
    n3.bNoMore = false;

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = p->BR;
    n4.vKeys.reserve(p->vKeys.size());
    n4.bNoMore = false;

    //Associate points to childs
    for (size_t i = 0; i < p->vKeys.size(); i++)
    {
        const cv::KeyPoint& kp = p->vKeys[i];

        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

static std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& N, const int& level, int nfeatures)
{
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    std::list<QuadTreeNode> lNodes;

    std::vector<QuadTreeNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++)
    {
        QuadTreeNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.bNoMore = false;
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    std::list<QuadTreeNode>::iterator lit = lNodes.begin();

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

    std::vector<std::pair<int, QuadTreeNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

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
                QuadTreeNode n1, n2, n3, n4;
                DivideNode(&*lit, n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0)
                {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0)
                {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0)
                {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0)
                {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
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

                std::vector<std::pair<int, QuadTreeNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                {
                    QuadTreeNode n1, n2, n3, n4;
                    DivideNode(vPrevSizeAndPointerToNode[j].second, n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            vSizeAndPointerToNode.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
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
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (auto lit = lNodes.begin(); lit != lNodes.end(); lit++)
    {
        std::vector<cv::KeyPoint>& vNodeKeys   = lit->vKeys;
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

static void IC_Angle(const cv::Mat& image, cv::KeyPoint &kp, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    kp.angle = 180.0 * atan2((float)m_01, (float)m_10) / M_PI;
}

/**
* 1. Splits every level of the image into evenly sized cells
* 2. Detects corners in a 7x7 cell area
* 3. Make sure key points are well distributed
* 4. Compute orientation of keypoints
* @param allKeypoints
*/
void KPExtractOrbSlam(std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat> &image_pyramid, PyramidParameters &p, float iniThFAST, float minThFAST)
{
    allKeypoints.resize(image_pyramid.size());
    std::vector<int> umax;
    init_patch(umax);

    const float W = 30;

    for (int level = 0; level < image_pyramid.size(); ++level)
    {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = image_pyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = image_pyramid[level].rows - EDGE_THRESHOLD + 3;

        std::vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(p.total_features * 10);

        const float width  = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float       maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = (float)maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float       maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = (float)maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                cv::FAST(image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                        vKeysCell, iniThFAST, true);


                if (vKeysCell.empty())
                {
                    cv::FAST(image_pyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                             vKeysCell, minThFAST, true);
                }

                if (!vKeysCell.empty())
                {
                    for (auto vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                    {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(p.total_features);

        keypoints = DistributeQuadTree(vToDistributeKeys,
                                       minBorderX,
                                       maxBorderX,
                                       minBorderY,
                                       maxBorderY,
                                       p.nb_feature_per_level[level],
                                       level,
                                       p.total_features);

        const float scaledPatchSize = (float)PATCH_SIZE * p.scale_factors[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;

            IC_Angle(image_pyramid[level], keypoints[i], umax);
            keypoints[i].pt *= p.scale_factors[level]; //Set position relative to level 0 image
        }
    }
}

// TILDE


class Parallel_process : public cv::ParallelLoopBody {

private:
    std::vector<float> &_param;
    std::vector<float> &_bias;
    std::vector<std::vector<float>> &_coeffs;
    std::vector<cv::Mat> &_filters;
    std::vector<cv::Mat> &_curRes;
    const int _nbApproximatedFilters;
    const std::vector<cv::Mat> &_vectorInput;

public:
    Parallel_process(const std::vector<cv::Mat> &conv, const int nb, std::vector<cv::Mat> &v, 
                     std::vector<float> &param, std::vector<float> &bias, 
                     std::vector<std::vector<float>> &coeffs, std::vector<cv::Mat> &filters)
    :
        _nbApproximatedFilters(nb),
        _vectorInput(conv),
        _curRes(v),
        _param(param),
        _bias(bias),
        _coeffs(coeffs),
        _filters(filters)
    {
    } 

    virtual void operator() (const cv::Range & range)const {

        for (int idxFilter = range.start; idxFilter < range.end; idxFilter++) {

            cv::Mat kernelX = _filters[idxFilter * 2 + 1];	// IMPORTANT!
            cv::Mat kernelY = _filters[idxFilter * 2];

            // the channel this filter is supposed to be applied to
            const int idxDim = idxFilter / _nbApproximatedFilters;
            cv::Mat res;
            cv::sepFilter2D(_vectorInput[idxDim], res, CV_32F, kernelX, kernelY, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
            _curRes[idxFilter] = res.clone(); // not cloning causes wierd issues.
        }
    }
};

std::vector<std::vector<cv::Mat>> getScoresForApprox(std::vector<float> &param, std::vector<float> &bias,
                                                     std::vector<std::vector<float>> &coeffs, 
                                                     std::vector<cv::Mat> &filters,
                                                     const std::vector<cv::Mat> &vectorInput)
{
    std::vector<std::vector<cv::Mat>>res;
    int nbMax = param[1]; 
    int nbSum = param[2];
    int nbOriginalFilters = nbMax * nbSum;
    int nbApproximatedFilters = param[3];
    int nbChannels = param[4];
    int sizeFilters = param[5];

    // allocate res
    res.resize(nbSum);
    for (int idxSum = 0; idxSum < nbSum; ++idxSum) {
        res[idxSum].resize(nbMax);
    }

    // calculate separable responses
    int idxSum = 0;
    int idxMax = 0;

    std::vector<cv::Mat>curRes((int)filters.size() / 2, cv::Mat(vectorInput[0].size(), CV_32F));	// temp storage

    cv::parallel_for_(cv::Range(0, (int)filters.size() / 2),
            Parallel_process(vectorInput, nbApproximatedFilters, curRes, param, bias, coeffs, filters));

    for (int idxFilter = 0; idxFilter < filters.size() / 2; idxFilter++) {
        //int idxOrig = 0;
        for (int idxOrig = 0; idxOrig < nbSum * nbMax; ++idxOrig) {
            int idxSum = idxOrig / nbMax;
            int idxMax = idxOrig % nbMax;

            if (idxFilter == 0) {
                res[idxSum][idxMax] = coeffs[idxOrig][idxFilter] * curRes[idxFilter].clone();
            } else {
                res[idxSum][idxMax] = res[idxSum][idxMax] + coeffs[idxOrig][idxFilter] * curRes[idxFilter];
            }

        }
    }

    // add the bias
    int idxOrig = 0;
    for (int idxSum = 0; idxSum < nbSum; ++idxSum) {
        for (int idxMax = 0; idxMax < nbMax; ++idxMax) {
            res[idxSum][idxMax] += bias[idxOrig];
            idxOrig++;
        }
    }

    return res;
}


void getCombinedScore(const std::vector<std::vector<cv::Mat>>& cascade_responses, cv::Mat *output)
{
    for (int idxCascade = 0; idxCascade < cascade_responses.size(); ++idxCascade)
    {
        cv::Mat respImageCascade = cascade_responses[idxCascade][0];

        for (int idxDepth = 1; idxDepth < cascade_responses[idxCascade].size(); ++idxDepth)
            respImageCascade = cv::max(respImageCascade, cascade_responses[idxCascade][idxDepth]);

        respImageCascade = idxCascade % 2 == 0 ? -respImageCascade : respImageCascade;
        if (idxCascade == 0)
            *output = respImageCascade;
        else
            *output = respImageCascade + *output;
    }

    //post process
    const float stdv = 2;
    const int sizeSmooth = 5 * stdv * 2 + 1;
    cv::GaussianBlur(*output, *output, cv::Size(sizeSmooth, sizeSmooth), stdv, stdv);
}


void KPExtractTILDE(std::vector<cv::KeyPoint>& allKeypoints, cv::Mat image)
{
    std::vector<float> param;
    std::vector<float> bias;
    std::vector<std::vector<float>> coeffs;
    std::vector<cv::Mat> filters;
    std::vector<std::string> tokens;

    cv::Mat im_resized;
    std::vector<cv::Mat> vectorInput;
    std::vector<std::vector<cv::Mat>>cascade_responses;
    cv::Mat outputScore;
    float resizeRatio;

    filters_open("filters/Chamonix24.txt", param, bias, coeffs, filters, tokens);
    resizeRatio = param[0];

    cv::resize(image, im_resized, cv::Size(0, 0), resizeRatio, resizeRatio);

    std::vector<cv::Mat> grad = image_gradient(im_resized);
    std::vector<cv::Mat> luv = rgb_to_luv(im_resized);

    std::copy(grad.begin(), grad.end(), std::back_inserter(vectorInput));
    std::copy(luv.begin(), luv.end(), std::back_inserter(vectorInput));

    cascade_responses = getScoresForApprox(param, bias, coeffs, filters, vectorInput);

    getCombinedScore(cascade_responses, &outputScore);

    std::vector<cv::Point3f> res_with_score = NonMaxSup(outputScore);

    // resize back
    resizeRatio = 1. / resizeRatio;

    for (int i = 0; i < res_with_score.size(); i++) 
    {
        cv::KeyPoint kp = cv::KeyPoint(res_with_score[i].x * resizeRatio, res_with_score[i].y * resizeRatio, 1.0, 0, res_with_score[i].z, 0);
        //cv::KeyPoint kp = cv::KeyPoint(res_with_score[i].x * resizeRatio, res_with_score[i].y * resizeRatio, 1.0, 0, 1, 0);
        kp.size = PATCH_SIZE;
        allKeypoints.push_back(kp);
    }
}

void KPExtractSURF(std::vector<cv::KeyPoint>& allKeypoints, cv::Mat image)
{
    cv::Ptr<cv::xfeatures2d::SURF> surf_detector = cv::xfeatures2d::SURF::create(1000);
    surf_detector->detect(image, allKeypoints);
}


