#include <WAIMapStorage.h>
#include <Profiler.h>

cv::Mat WAIMapStorage::convertToCVMat(const SLMat4f slMat)
{
    cv::Mat cvMat = cv::Mat(4, 4, CV_32F);
    // clang-format off
        //so ein scheiss!!!
        //  T M0, T M4, T M8, T M12,
        //  T M1, T M5, T M9, T M13,
        //  T M2, T M6, T M10, T M14,
        //  T M3, T M7, T M11, T M15)
    cvMat.at<float>(0, 0) = slMat.m(0);  
    cvMat.at<float>(1, 0) = slMat.m(1); 
    cvMat.at<float>(2, 0) = slMat.m(2);  
    cvMat.at<float>(3, 0) = slMat.m(3);

    cvMat.at<float>(0, 1) = slMat.m(4);
    cvMat.at<float>(1, 1) = slMat.m(5);
    cvMat.at<float>(2, 1) = slMat.m(6);
    cvMat.at<float>(3, 1) = slMat.m(7);

    cvMat.at<float>(0, 2) = slMat.m(8);
    cvMat.at<float>(1, 2) = slMat.m(9);
    cvMat.at<float>(2, 2) = slMat.m(10);
    cvMat.at<float>(3, 2) = slMat.m(11);

    cvMat.at<float>(0, 3) = slMat.m(12);
    cvMat.at<float>(1, 3) = slMat.m(13);
    cvMat.at<float>(2, 3) = slMat.m(14);
    cvMat.at<float>(3, 3) = slMat.m(15);
    // clang-format on
    return cvMat;
}

SLMat4f WAIMapStorage::convertToSLMat(const cv::Mat& cvMat)
{
    SLMat4f slMat;
    // clang-format off
        //  T M0, T M4, T M8, T M12,
        //  T M1, T M5, T M9, T M13,
        //  T M2, T M6, T M10, T M14,
        //  T M3, T M7, T M11, T M15)
    slMat.setMatrix(
        cvMat.at<float>(0, 0), cvMat.at<float>(0, 1), cvMat.at<float>(0, 2), cvMat.at<float>(0, 3), 
        cvMat.at<float>(1, 0), cvMat.at<float>(1, 1), cvMat.at<float>(1, 2), cvMat.at<float>(1, 3), 
        cvMat.at<float>(2, 0), cvMat.at<float>(2, 1), cvMat.at<float>(2, 2), cvMat.at<float>(2, 3), 
        cvMat.at<float>(3, 0), cvMat.at<float>(3, 1), cvMat.at<float>(3, 2), cvMat.at<float>(3, 3));
    // clang-format on

    return slMat;
}

void buildMatching(std::vector<WAIKeyFrame*>&                        kfs,
                   std::map<WAIKeyFrame*, std::map<size_t, size_t>>& KFmatching)
{
    for (int i = 0; i < kfs.size(); ++i)
    {
        WAIKeyFrame* kf = kfs[i];
        if (kf->isBad())
            continue;
        // if (kf->mBowVec.data.empty())
        //     continue;

        std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
        std::map<size_t, size_t>  matching;

        size_t id = 0;
        for (int j = 0; j < mps.size(); j++)
        {
            if (mps[j] != nullptr)
            {
                matching.insert(std::pair<size_t, size_t>(j, id));
                id++;
            }
        }
        KFmatching.insert(std::pair<WAIKeyFrame*, std::map<size_t, size_t>>(kf, matching));
    }
}

void saveKeyFrames(std::vector<WAIKeyFrame*>&                        kfs,
                   std::map<WAIKeyFrame*, std::map<size_t, size_t>>& KFmatching,
                   cv::FileStorage&                                  fs,
                   std::string                                       imgDir,
                   bool                                              saveBOW)
{
    // start sequence keyframes
    fs << "KeyFrames"
       << "[";
    for (int i = 0; i < kfs.size(); ++i)
    {
        WAIKeyFrame* kf = kfs[i];
        if (kf->isBad())
            continue;
        if (kf->mBowVec.data.empty())
            continue;

        fs << "{";         // new map keyFrame
                           // add id
        fs << "id" << (int)kf->mnId;
        if (kf->mnId != 0) // kf with id 0 has no parent
            fs << "parentId" << (int)kf->GetParent()->mnId;
        else
            fs << "parentId" << -1;

        // loop edges: we store the id of the connected kf
        auto loopEdges = kf->GetLoopEdges();
        if (loopEdges.size())
        {
            std::vector<int> loopEdgeIds;
            for (auto loopEdgeKf : loopEdges)
            {
                loopEdgeIds.push_back(loopEdgeKf->mnId);
            }
            fs << "loopEdges" << loopEdgeIds;
        }

        // world w.r.t camera
        fs << "Tcw" << kf->GetPose();

        if (KFmatching.size() > 0)
        {
            cv::Mat                         descriptors;
            const std::map<size_t, size_t>& matching = KFmatching[kf];
            descriptors.create((int)matching.size(), 32, CV_8U);
            std::vector<cv::KeyPoint> keypoints(matching.size());
            for (int j = 0; j < kf->mvKeysUn.size(); j++)
            {
                auto it = matching.find(j);
                if (it != matching.end())
                {
                    kf->mDescriptors.row(j).copyTo(descriptors.row(it->second));
                    keypoints[it->second] = kf->mvKeysUn[j];
                }
            }
            fs << "featureDescriptors" << descriptors;
            fs << "keyPtsUndist" << keypoints;
        }
        else
        {
            fs << "featureDescriptors" << kf->mDescriptors;
            fs << "keyPtsUndist" << kf->mvKeysUn;
        }

        if (saveBOW)
        {
            WAIBowVector&      bowVec = kf->mBowVec;
            std::vector<int>   wordsId;
            std::vector<float> tfIdf;
            for (auto it = bowVec.getWordScoreMapping().begin(); it != bowVec.getWordScoreMapping().end(); it++)
            {
                wordsId.push_back(it->first);
                tfIdf.push_back(it->second);
            }

            fs << "BowVectorWordsId" << wordsId;
            fs << "TfIdf" << tfIdf;
        }

        // scale factor
        fs << "scaleFactor" << kf->mfScaleFactor;
        // number of pyriamid scale levels
        fs << "nScaleLevels" << kf->mnScaleLevels;
        fs << "K" << kf->mK;

        fs << "nMinX" << kf->mnMinX;
        fs << "nMinY" << kf->mnMinY;
        fs << "nMaxX" << kf->mnMaxX;
        fs << "nMaxY" << kf->mnMaxY;

#if 0
        std::vector<int>          bestCovisibleKeyFrameIds;
        std::vector<int>          bestCovisibleWeights;
        std::vector<WAIKeyFrame*> bestCovisibles = kf->GetBestCovisibilityKeyFrames(20);
        for (WAIKeyFrame* covisible : bestCovisibles)
        {
            if (covisible->isBad())
                continue;
            int weight = kf->GetWeight(covisible);
            if (weight)
            {
                bestCovisibleKeyFrameIds.push_back(covisible->mnId);
                bestCovisibleWeights.push_back(weight);
            }
        }

        fs << "bestCovisibleKeyFrameIds" << bestCovisibleKeyFrameIds;
        fs << "bestCovisibleWeights" << bestCovisibleWeights;
#endif

        fs << "}"; // close map

        // save the original frame image for this keyframe
        if (imgDir != "")
        {
            cv::Mat imgColor;
            if (!kf->imgGray.empty())
            {
                std::stringstream ss;
                ss << imgDir << "kf" << (int)kf->mnId << ".jpg";

                cv::cvtColor(kf->imgGray, imgColor, cv::COLOR_GRAY2BGR);
                cv::imwrite(ss.str(), imgColor);

                // if this kf was never loaded, we still have to set the texture path
                kf->setTexturePath(ss.str());
            }
        }
    }
    fs << "]"; // close sequence keyframes
}

void saveMapPoints(std::vector<WAIMapPoint*>                         mpts,
                   std::map<WAIKeyFrame*, std::map<size_t, size_t>>& KFmatching,
                   cv::FileStorage&                                  fs)
{
    // start map points sequence
    fs << "MapPoints"
       << "[";
    for (int i = 0; i < mpts.size(); ++i)
    {
        WAIMapPoint* mpt = mpts[i];
        // TODO: ghm1: check if it is necessary to removed points that have no reference keyframe OR can we somehow update the reference keyframe in the SLAM
        if (mpt->isBad() || mpt->refKf()->isBad())
            continue;

        fs << "{"; // new map for MapPoint
                   // add id
        fs << "id" << (int)mpt->mnId;
        // add position
        fs << "mWorldPos" << mpt->GetWorldPos();
        // save keyframe observations
        auto        observations = mpt->GetObservations();
        vector<int> observingKfIds;
        vector<int> corrKpIndices; // corresponding keypoint indices in observing keyframe

        if (!KFmatching.empty())
        {
            for (auto it : observations)
            {
                WAIKeyFrame* kf    = it.first;
                size_t       kpIdx = it.second;
                if (!kf || kf->isBad() || kf->mBowVec.data.empty())
                    continue;

                if (KFmatching.find(kf) == KFmatching.end())
                {
                    std::cout << "observation not found in kfmatching" << std::endl;
                    continue;
                }

                const std::map<size_t, size_t>& matching = KFmatching[kf];
                auto                            mit      = matching.find(kpIdx);
                if (mit != matching.end())
                {
                    observingKfIds.push_back(kf->mnId);
                    corrKpIndices.push_back(mit->second);
                }
            }
        }
        else
        {
            for (auto it : observations)
            {
                if (!it.first->isBad())
                {
                    observingKfIds.push_back(it.first->mnId);
                    corrKpIndices.push_back(it.second);
                }
            }
        }

        fs << "observingKfIds" << observingKfIds;
        fs << "corrKpIndices" << corrKpIndices;

        fs << "mfMaxDistance" << mpt->GetMaxDistance();
        fs << "mfMinDistance" << mpt->GetMinDistance();
        fs << "mNormalVector" << mpt->GetNormal();
        fs << "mDescriptor" << mpt->GetDescriptor();

        fs << "refKfId" << (int)mpt->refKf()->mnId;
        fs << "}"; // close map
    }
    fs << "]";
}

bool WAIMapStorage::saveMap(WAIMap*     waiMap,
                            SLNode*     mapNode,
                            std::string filename,
                            std::string imgDir,
                            bool        saveBOW)
{
    std::vector<WAIKeyFrame*>                        kfs  = waiMap->GetAllKeyFrames();
    std::vector<WAIMapPoint*>                        mpts = waiMap->GetAllMapPoints();
    std::map<WAIKeyFrame*, std::map<size_t, size_t>> KFmatching;

    if (kfs.size() == 0)
        return false;

    buildMatching(kfs, KFmatching);

    // save keyframes (without graph/neigbourhood information)
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        return false;
    }

    if (mapNode)
    {
        SLMat4f slOm = mapNode->om();
        std::cout << "slOm: " << slOm.toString() << std::endl;
        cv::Mat cvOm = convertToCVMat(mapNode->om());
        std::cout << "cvOM: " << cvOm << std::endl;
        fs << "mapNodeOm" << cvOm;
    }

    saveKeyFrames(kfs, KFmatching, fs, imgDir, saveBOW);
    saveMapPoints(mpts, KFmatching, fs);

    // explicit close
    fs.release();
    return true;
}

bool WAIMapStorage::saveMapRaw(WAIMap*     waiMap,
                               SLNode*     mapNode,
                               std::string filename,
                               std::string imgDir)
{
    std::vector<WAIKeyFrame*>                        kfs  = waiMap->GetAllKeyFrames();
    std::vector<WAIMapPoint*>                        mpts = waiMap->GetAllMapPoints();
    std::map<WAIKeyFrame*, std::map<size_t, size_t>> KFmatching;

    if (kfs.size() == 0)
        return false;

    // in this case we dont build a keyframe matching..

    // save keyframes (without graph/neigbourhood information)
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    if (!fs.isOpened())
    {
        return false;
    }

    if (mapNode)
    {
        SLMat4f slOm = mapNode->om();
        std::cout << "slOm: " << slOm.toString() << std::endl;
        cv::Mat cvOm = convertToCVMat(mapNode->om());
        std::cout << "cvOM: " << cvOm << std::endl;
        fs << "mapNodeOm" << cvOm;
    }

    saveKeyFrames(kfs, KFmatching, fs, imgDir, false);
    saveMapPoints(mpts, KFmatching, fs);

    // explicit close
    fs.release();
    return true;
}

std::vector<uint8_t> WAIMapStorage::convertCVMatToVector(const cv::Mat& mat)
{
    std::vector<uint8_t> result;

    // makes sure mat is continuous
    cv::Mat continuousMat = mat.clone();

    // TODO(dgj1): verify that this is correct
    result.assign(continuousMat.data, continuousMat.data + continuousMat.total() * continuousMat.elemSize());

    return result;
}

template<typename T>
void WAIMapStorage::writeVectorToBinaryFile(FILE* f, const std::vector<T> vec)
{
    fwrite(vec.data(), sizeof(T), vec.size(), f);
}

void WAIMapStorage::writeCVMatToBinaryFile(FILE* f, const cv::Mat& mat)
{
    std::vector<uint8_t> data = convertCVMatToVector(mat);

    writeVectorToBinaryFile(f, data);
}

bool WAIMapStorage::saveMapBinary(WAIMap*     waiMap,
                                  SLNode*     mapNode,
                                  std::string filename,
                                  std::string imgDir,
                                  bool        saveBOW)
{
    std::vector<WAIKeyFrame*>                        kfs  = waiMap->GetAllKeyFrames();
    std::vector<WAIMapPoint*>                        mpts = waiMap->GetAllMapPoints();
    std::map<WAIKeyFrame*, std::map<size_t, size_t>> KFmatching;

    if (kfs.size() == 0)
        return false;

    buildMatching(kfs, KFmatching);

    FILE* f = fopen(filename.c_str(), "wb");
    if (!f)
    {
        return false;
    }

    WAIMapStorage::MapInfo mapInfo = {};
    mapInfo.version                = 1;

    for (int i = 0; i < kfs.size(); ++i)
    {
        WAIKeyFrame* kf = kfs[i];
        if (kf->isBad())
            continue;
        if (kf->mBowVec.data.empty())
            continue;

        mapInfo.kfCount++;
    }

    for (int i = 0; i < mpts.size(); ++i)
    {
        WAIMapPoint* mpt = mpts[i];
        // TODO: ghm1: check if it is necessary to removed points that have no reference keyframe OR can we somehow update the reference keyframe in the SLAM
        if (mpt->isBad() || mpt->refKf()->isBad())
            continue;

        mapInfo.mpCount++;
    }

    if (mapNode)
    {
        mapInfo.nodeOmSaved = true;
    }

    fwrite(&mapInfo, sizeof(WAIMapStorage::MapInfo), 1, f);

    std::vector<uchar> omVec;
    if (mapNode)
    {
        SLMat4f slOm = mapNode->om();
        cv::Mat cvOm = convertToCVMat(mapNode->om());

        writeCVMatToBinaryFile(f, cvOm);
    }

    // start keyframes sequence
    for (int i = 0; i < kfs.size(); ++i)
    {
        WAIKeyFrame* kf = kfs[i];
        if (kf->isBad())
            continue;
        if (kf->mBowVec.data.empty())
            continue;

        WAIMapStorage::KeyFrameInfo kfInfo = {};

        kfInfo.id = (int32_t)kf->mnId;

        // scale factor
        kfInfo.scaleFactor = kf->mfScaleFactor;
        // number of pyramid scale levels
        kfInfo.scaleLevels = kf->mnScaleLevels;

        kfInfo.minX = kf->mnMinX;
        kfInfo.minY = kf->mnMinY;
        kfInfo.maxX = kf->mnMaxX;
        kfInfo.maxY = kf->mnMaxY;

        cv::Mat     descriptors;
        CVVKeyPoint keyPoints;
        if (KFmatching.size() > 0)
        {
            const std::map<size_t, size_t>& matching = KFmatching[kf];
            descriptors.create((int)matching.size(), 32, CV_8U);
            keyPoints.resize(matching.size());
            for (int j = 0; j < kf->mvKeysUn.size(); j++)
            {
                auto it = matching.find(j);
                if (it != matching.end())
                {
                    kf->mDescriptors.row(j).copyTo(descriptors.row(it->second));
                    keyPoints[it->second] = kf->mvKeysUn[j];
                }
            }
        }
        else
        {
            descriptors = kf->mDescriptors;
            keyPoints   = kf->mvKeysUn;
        }

        kfInfo.kpCount = keyPoints.size();

        if (kf->mnId != 0) // kf with id 0 has no parent
            kfInfo.parentId = (int32_t)kf->GetParent()->mnId;
        else
            kfInfo.parentId = -1;

        // loop edges: we store the id of the connected kf
        std::set<WAIKeyFrame*> loopEdges = kf->GetLoopEdges();
        std::vector<int32_t>   loopEdgeIds;
        if (loopEdges.size())
        {
            for (WAIKeyFrame* loopEdgeKf : loopEdges)
            {
                loopEdgeIds.push_back(loopEdgeKf->mnId);
                kfInfo.loopEdgesCount++; // TODO(dgj1): probably not ideal for cache coherence
            }
        }

        std::vector<int32_t> wordsId;
        std::vector<float>   tfIdf;
        if (saveBOW)
        {
            WAIBowVector& bowVec = kf->mBowVec;
            for (auto it = bowVec.getWordScoreMapping().begin();
                 it != bowVec.getWordScoreMapping().end();
                 it++)
            {
                wordsId.push_back(it->first);
                tfIdf.push_back(it->second);
            }

            kfInfo.bowVecSize = wordsId.size();
        }

        std::vector<int32_t>      bestCovisibleKeyFrameIds;
        std::vector<int32_t>      bestCovisibleWeights;
        std::vector<WAIKeyFrame*> bestCovisibles = kf->GetBestCovisibilityKeyFrames(20);
        for (WAIKeyFrame* covisible : bestCovisibles)
        {
            if (covisible->isBad())
                continue;
            int weight = kf->GetWeight(covisible);
            if (weight)
            {
                bestCovisibleKeyFrameIds.push_back(covisible->mnId);
                bestCovisibleWeights.push_back(weight);
            }
        }

        kfInfo.covisiblesCount = bestCovisibleKeyFrameIds.size();

        fwrite(&kfInfo, sizeof(KeyFrameInfo), 1, f);
        writeCVMatToBinaryFile(f, kf->mK);
        writeCVMatToBinaryFile(f, kf->GetPose());
        writeVectorToBinaryFile(f, loopEdgeIds);
        writeCVMatToBinaryFile(f, descriptors);
        writeVectorToBinaryFile(f, keyPoints);

        if (saveBOW)
        {
            writeVectorToBinaryFile(f, wordsId);
            writeVectorToBinaryFile(f, tfIdf);
        }

        writeVectorToBinaryFile(f, bestCovisibleKeyFrameIds);
        writeVectorToBinaryFile(f, bestCovisibleWeights);

        // save the original frame image for this keyframe
        if (imgDir != "")
        {
            cv::Mat imgColor;
            if (!kf->imgGray.empty())
            {
                std::stringstream ss;
                ss << imgDir << "kf" << (int)kf->mnId << ".jpg";

                cv::cvtColor(kf->imgGray, imgColor, cv::COLOR_GRAY2BGR);
                cv::imwrite(ss.str(), imgColor);

                // if this kf was never loaded, we still have to set the texture path
                kf->setTexturePath(ss.str());
            }
        }
    }

    // start map points sequence
    for (int i = 0; i < mpts.size(); ++i)
    {
        WAIMapPoint* mpt = mpts[i];
        // TODO: ghm1: check if it is necessary to removed points that have no reference keyframe OR can we somehow update the reference keyframe in the SLAM
        if (mpt->isBad() || mpt->refKf()->isBad())
            continue;

        MapPointInfo mpInfo = {};
        mpInfo.id           = (int32_t)mpt->mnId;
        mpInfo.refKfId      = (int32_t)mpt->refKf()->mnId;

        // save keyframe observations
        std::map<WAIKeyFrame*, size_t> observations = mpt->GetObservations();
        vector<int32_t>                observingKfIds;
        vector<int32_t>                corrKpIndices; // corresponding keypoint indices in observing keyframe
        if (!KFmatching.empty())
        {
            for (std::pair<WAIKeyFrame* const, size_t> it : observations)
            {
                WAIKeyFrame* kf    = it.first;
                size_t       kpIdx = it.second;
                if (!kf || kf->isBad() || kf->mBowVec.data.empty())
                    continue;

                if (KFmatching.find(kf) == KFmatching.end())
                {
                    std::cout << "observation not found in kfmatching" << std::endl;
                    continue;
                }

                const std::map<size_t, size_t>& matching = KFmatching[kf];
                auto                            mit      = matching.find(kpIdx);
                if (mit != matching.end())
                {
                    observingKfIds.push_back(kf->mnId);
                    corrKpIndices.push_back(mit->second);
                }
            }
        }
        else
        {
            for (std::pair<WAIKeyFrame* const, size_t> it : observations)
            {
                if (!it.first->isBad())
                {
                    observingKfIds.push_back(it.first->mnId);
                    corrKpIndices.push_back(it.second);
                }
            }
        }

        mpInfo.nObervations = observingKfIds.size();
        mpInfo.minDistance  = mpt->GetMinDistance();
        mpInfo.maxDistance  = mpt->GetMaxDistance();

        fwrite(&mpInfo, sizeof(mpInfo), 1, f);
        writeCVMatToBinaryFile(f, mpt->GetWorldPos());
        writeVectorToBinaryFile(f, observingKfIds);
        writeVectorToBinaryFile(f, corrKpIndices);
        writeCVMatToBinaryFile(f, mpt->GetNormal());
        writeCVMatToBinaryFile(f, mpt->GetDescriptor());
    }

    fclose(f);

    return true;
}

template<typename T>
std::vector<T> WAIMapStorage::loadVectorFromBinaryStream(uint8_t** data, int count)
{
    std::vector<T> result((T*)(*data), ((T*)(*data)) + count);
    *data += sizeof(T) * count;

    return result;
}

// returns the number of bytes loaded
cv::Mat WAIMapStorage::loadCVMatFromBinaryStream(uint8_t** data,
                                                 int       rows,
                                                 int       cols,
                                                 int       type)
{
    cv::Mat result = cv::Mat(rows, cols, type, *data);

    *data += rows * cols * result.elemSize();

    return result;
}

bool WAIMapStorage::loadMapBinary(WAIMap*           waiMap,
                                  cv::Mat&          mapNodeOm,
                                  WAIOrbVocabulary* voc,
                                  std::string       path,
                                  bool              loadImgs,
                                  bool              fixKfsAndMPts)
{
    PROFILE_FUNCTION();

    std::vector<WAIMapPoint*>       mapPoints;
    std::vector<WAIKeyFrame*>       keyFrames;
    std::map<int, int>              parentIdMap;
    std::map<int, std::vector<int>> loopEdgesMap;
    std::map<int, WAIKeyFrame*>     kfsMap;
    int                             numLoopClosings = 0;

    std::string imgDir;
    if (loadImgs)
    {
        std::string dir = Utils::getPath(path);
        imgDir          = dir + Utils::getFileNameWOExt(path) + "/";
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;

    fseek(f, 0, SEEK_END);
    uint32_t contentSize = ftell(f);
    rewind(f);

    uint8_t* fContent      = (uint8_t*)malloc(sizeof(uint8_t*) * contentSize);
    uint8_t* fContentStart = fContent;
    if (!fContent)
        return false;

    size_t readResult = fread(fContent, 1, contentSize, f);
    if (readResult != contentSize)
        return false;

    fclose(f);

    MapInfo* mapInfo = (MapInfo*)fContent;
    fContent += sizeof(MapInfo);

    if (mapInfo->nodeOmSaved)
    {
        cv::Mat cvMat = loadCVMatFromBinaryStream(&fContent, 4, 4, CV_32F);
        mapNodeOm     = cvMat.clone();
    }

    std::map<int, std::vector<int>> bestCovisibleKeyFrameIdsMap;
    std::map<int, std::vector<int>> bestCovisibleWeightsMap;

    for (int i = 0; i < mapInfo->kfCount; i++)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::keyFrames");

        KeyFrameInfo* kfInfo = (KeyFrameInfo*)fContent;
        fContent += sizeof(KeyFrameInfo);

        int id       = kfInfo->id;
        int parentId = kfInfo->parentId;

        if (parentId != -1)
        {
            parentIdMap[id] = parentId;
        }

        cv::Mat K   = loadCVMatFromBinaryStream(&fContent, 3, 3, CV_32F);
        cv::Mat Tcw = loadCVMatFromBinaryStream(&fContent, 4, 4, CV_32F);

        if (kfInfo->loopEdgesCount > 0)
        {
            std::vector<int32_t> loopEdges =
              loadVectorFromBinaryStream<int32_t>(&fContent, kfInfo->loopEdgesCount);

            loopEdgesMap[id] = loopEdges;
        }

        cv::Mat                   featureDescriptors = loadCVMatFromBinaryStream(&fContent, kfInfo->kpCount, 32, CV_8U);
        std::vector<cv::KeyPoint> keyPtsUndist       = loadVectorFromBinaryStream<cv::KeyPoint>(&fContent, kfInfo->kpCount);

        float scaleFactor  = kfInfo->scaleFactor;
        int   nScaleLevels = kfInfo->scaleLevels;

        // vectors for precalculation of scalefactors
        std::vector<float> vScaleFactor;
        std::vector<float> vInvScaleFactor;
        std::vector<float> vLevelSigma2;
        std::vector<float> vInvLevelSigma2;
        vScaleFactor.clear();
        vLevelSigma2.clear();
        vScaleFactor.resize(nScaleLevels);
        vLevelSigma2.resize(nScaleLevels);
        // todo:  crashes when vScaleFactor is empty
        vScaleFactor[0] = 1.0f;
        vLevelSigma2[0] = 1.0f;
        for (int j = 1; j < nScaleLevels; j++)
        {
            vScaleFactor[j] = vScaleFactor[j - 1] * scaleFactor;
            vLevelSigma2[j] = vScaleFactor[j] * vScaleFactor[j];
        }

        vInvScaleFactor.resize(nScaleLevels);
        vInvLevelSigma2.resize(nScaleLevels);
        for (int j = 0; j < nScaleLevels; j++)
        {
            vInvScaleFactor[j] = 1.0f / vScaleFactor[j];
            vInvLevelSigma2[j] = 1.0f / vLevelSigma2[j];
        }

        float fx, fy, cx, cy;
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);

        // image bounds
        float nMinX = kfInfo->minX;
        float nMinY = kfInfo->minY;
        float nMaxX = kfInfo->maxX;
        float nMaxY = kfInfo->maxY;

        WAIKeyFrame* newKf = new WAIKeyFrame(Tcw,
                                             id,
                                             fixKfsAndMPts,
                                             fx,
                                             fy,
                                             cx,
                                             cy,
                                             keyPtsUndist.size(),
                                             keyPtsUndist,
                                             featureDescriptors,
                                             voc,
                                             nScaleLevels,
                                             scaleFactor,
                                             vScaleFactor,
                                             vLevelSigma2,
                                             vInvLevelSigma2,
                                             (int)nMinX,
                                             (int)nMinY,
                                             (int)nMaxX,
                                             (int)nMaxY,
                                             K);

        if (kfInfo->bowVecSize > 0)
        {
            std::vector<int32_t> wordsId = loadVectorFromBinaryStream<int32_t>(&fContent, kfInfo->bowVecSize);
            std::vector<float>   tfIdf   = loadVectorFromBinaryStream<float>(&fContent, kfInfo->bowVecSize);

            WAIBowVector bow(wordsId, tfIdf);
            newKf->SetBowVector(bow);
        }

        if (imgDir != "")
        {
            stringstream ss;
            ss << imgDir << "kf" << id << ".jpg";
            // newKf->imgGray = kfImg;
            if (Utils::fileExists(ss.str()))
            {
                newKf->setTexturePath(ss.str());
                cv::Mat imgColor = cv::imread(ss.str());
                cv::cvtColor(imgColor, newKf->imgGray, cv::COLOR_BGR2GRAY);
            }
        }

        keyFrames.push_back(newKf);
        kfsMap[newKf->mnId] = newKf;

        std::vector<int32_t> bestCovisibleKeyFrameIds = loadVectorFromBinaryStream<int32_t>(&fContent, kfInfo->covisiblesCount);
        std::vector<int32_t> bestCovisibleWeights     = loadVectorFromBinaryStream<int32_t>(&fContent, kfInfo->covisiblesCount);

        bestCovisibleKeyFrameIdsMap[newKf->mnId] = bestCovisibleKeyFrameIds;
        bestCovisibleWeightsMap[newKf->mnId]     = bestCovisibleWeights;
    }

    // set parent keyframe pointers into keyframes
    for (WAIKeyFrame* kf : keyFrames)
    {
        if (kf->mnId != 0)
        {
            auto itParentId = parentIdMap.find(kf->mnId);
            if (itParentId != parentIdMap.end())
            {
                int  parentId   = itParentId->second;
                auto itParentKf = kfsMap.find(parentId);
                if (itParentKf != kfsMap.end())
                    kf->ChangeParent(itParentKf->second);
                else
                    cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist of keyframe " << kf->mnId << "! FAIL" << endl;
            }
            else
                cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist of keyframe " << kf->mnId << "! FAIL" << endl;
        }
    }

    int numberOfLoopClosings = 0;
    // set loop edge pointer into keyframes
    for (WAIKeyFrame* kf : keyFrames)
    {
        auto it = loopEdgesMap.find(kf->mnId);
        if (it != loopEdgesMap.end())
        {
            const auto& loopEdgeIds = it->second;
            for (int loopKfId : loopEdgeIds)
            {
                auto loopKfIt = kfsMap.find(loopKfId);
                if (loopKfIt != kfsMap.end())
                {
                    kf->AddLoopEdge(loopKfIt->second);
                    numberOfLoopClosings++;
                }
                else
                    cerr << "[WAIMapIO] loadKeyFrames: Loop keyframe id does not exist! FAIL" << endl;
            }
        }
    }
    numLoopClosings = numberOfLoopClosings / 2;

    for (int i = 0; i < mapInfo->mpCount; i++)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::mapPoints");

        MapPointInfo* mpInfo = (MapPointInfo*)fContent;
        fContent += sizeof(MapPointInfo);

        int id = mpInfo->id;

        cv::Mat              mWorldPos      = loadCVMatFromBinaryStream(&fContent, 3, 1, CV_32F);
        std::vector<int32_t> observingKfIds = loadVectorFromBinaryStream<int32_t>(&fContent, mpInfo->nObervations);
        std::vector<int32_t> corrKpIndices  = loadVectorFromBinaryStream<int32_t>(&fContent, mpInfo->nObervations);

        cv::Mat normal     = loadCVMatFromBinaryStream(&fContent, 3, 1, CV_32F);
        cv::Mat descriptor = loadCVMatFromBinaryStream(&fContent, 1, 32, CV_8U);

        WAIMapPoint* newPt = new WAIMapPoint(id, mWorldPos, fixKfsAndMPts);
        newPt->SetMinDistance(mpInfo->minDistance);
        newPt->SetMaxDistance(mpInfo->maxDistance);
        newPt->SetNormal(normal);
        newPt->SetDescriptor(descriptor);

        // get reference keyframe id
        int  refKfId    = (int)mpInfo->refKfId;
        bool refKFFound = false;

        if (kfsMap.find(refKfId) != kfsMap.end())
        {
            newPt->refKf(kfsMap[refKfId]);
            refKFFound = true;
        }
        else
        {
            cout << "no reference keyframe found!" << endl;
            if (observingKfIds.size())
            {
                // we use the first of the observing keyframes
                int kfId = observingKfIds[0];
                if (kfsMap.find(kfId) != kfsMap.end())
                {
                    newPt->refKf(kfsMap[kfId]);
                    refKFFound = true;
                }
            }
        }

        if (refKFFound)
        {
            // find and add pointers of observing keyframes to map point
            for (int j = 0; j < observingKfIds.size(); j++)
            {
                const int kfId = observingKfIds[j];
                if (kfsMap.find(kfId) != kfsMap.end())
                {
                    WAIKeyFrame* kf = kfsMap[kfId];
                    kf->AddMapPoint(newPt, corrKpIndices[j]);
                    newPt->AddObservation(kf, corrKpIndices[j]);
                }
            }
            mapPoints.push_back(newPt);
        }
        else
        {
            delete newPt;
        }
    }

    // update the covisibility graph, when all keyframes and mappoints are loaded
    WAIKeyFrame* firstKF           = nullptr;
    bool         buildSpanningTree = false;
    for (WAIKeyFrame* kf : keyFrames)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::updateConnections");

        std::map<WAIKeyFrame*, int> keyFrameWeightMap;

        std::vector<int> bestCovisibleKeyFrameIds = bestCovisibleKeyFrameIdsMap[kf->mnId];
        std::vector<int> bestCovisibleWeights     = bestCovisibleWeightsMap[kf->mnId];

        for (int i = 0; i < bestCovisibleKeyFrameIds.size(); i++)
        {
            int          keyFrameId        = bestCovisibleKeyFrameIds[i];
            int          weight            = bestCovisibleWeights[i];
            WAIKeyFrame* covisibleKF       = kfsMap[keyFrameId];
            keyFrameWeightMap[covisibleKF] = weight;
        }

        kf->UpdateConnections(keyFrameWeightMap, false);

        if (kf->mnId == 0)
        {
            firstKF = kf;
        }
        else if (kf->GetParent() == NULL)
        {
            buildSpanningTree = true;
        }
    }

    wai_assert(firstKF && "Could not find keyframe with id 0\n");

    // Build spanning tree if keyframes have no parents (legacy support)
    if (buildSpanningTree)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::buildSpanningTree");

        // QueueElem: <unconnected_kf, graph_kf, weight>
        using QueueElem = std::tuple<WAIKeyFrame*, WAIKeyFrame*, int>;
        auto cmpQueue   = [](const QueueElem& left, const QueueElem& right)
        { return (std::get<2>(left) < std::get<2>(right)); };
        auto cmpMap = [](const pair<WAIKeyFrame*, int>& left, const pair<WAIKeyFrame*, int>& right)
        { return left.second < right.second; };
        std::set<WAIKeyFrame*> graph;
        std::set<WAIKeyFrame*> unconKfs;
        for (auto& kf : keyFrames)
            unconKfs.insert(kf);

        // pick first kf
        graph.insert(firstKF);
        unconKfs.erase(firstKF);

        while (unconKfs.size())
        {
            std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmpQueue)> q(cmpQueue);
            // update queue with keyframes with neighbous in the graph
            for (auto& unconKf : unconKfs)
            {
                const std::map<WAIKeyFrame*, int>& weights = unconKf->GetConnectedKfWeights();
                for (auto& graphKf : graph)
                {
                    auto it = weights.find(graphKf);
                    if (it != weights.end())
                    {
                        QueueElem newElem = std::make_tuple(unconKf, it->first, it->second);
                        q.push(newElem);
                    }
                }
            }

            if (q.size() == 0)
            {
                // no connection: the remaining keyframes are unconnected
                Utils::log("WAIMapStorage", "Error in building spanning tree: There are %i unconnected keyframes!", unconKfs.size());
                break;
            }
            else
            {
                // extract keyframe with shortest connection
                QueueElem topElem = q.top();
                // remove it from unconKfs and add it to graph
                WAIKeyFrame* newGraphKf = std::get<0>(topElem);
                unconKfs.erase(newGraphKf);
                newGraphKf->ChangeParent(std::get<1>(topElem));
                // std::cout << "Added kf " << newGraphKf->mnId << " with parent " << std::get<1>(topElem)->mnId << std::endl;
                // update parent
                graph.insert(newGraphKf);
            }
        }
    }

    for (WAIKeyFrame* kf : keyFrames)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::addKeyFrame");

        if (kf->mBowVec.data.empty())
        {
            std::cout << "kf->mBowVec.data empty" << std::endl;
            continue;
        }
        waiMap->AddKeyFrame(kf);
        waiMap->GetKeyFrameDB()->add(kf);

        // Add keyframe with id 0 to this vector. Otherwise RunGlobalBundleAdjustment in LoopClosing after loop was detected crashes.
        if (kf->mnId == 0)
        {
            waiMap->mvpKeyFrameOrigins.push_back(kf);
        }
    }

    for (WAIMapPoint* point : mapPoints)
    {
        PROFILE_SCOPE("WAI::WAIMapStorage::loadMapBinary::addMapPoint");

        waiMap->AddMapPoint(point);
    }

    waiMap->setNumLoopClosings(numLoopClosings);

    free(fContentStart);

    return true;
}

bool WAIMapStorage::loadMap(WAIMap*           waiMap,
                            cv::Mat&          mapNodeOm,
                            WAIOrbVocabulary* voc,
                            std::string       path,
                            bool              loadImgs,
                            bool              fixKfsAndMPts)
{
    PROFILE_FUNCTION();

    std::vector<WAIMapPoint*>       mapPoints;
    std::vector<WAIKeyFrame*>       keyFrames;
    std::map<int, int>              parentIdMap;
    std::map<int, std::vector<int>> loopEdgesMap;
    std::map<int, WAIKeyFrame*>     kfsMap;
    int                             numLoopClosings = 0;

    std::string imgDir;
    if (loadImgs)
    {
        std::string dir = Utils::getPath(path);
        imgDir          = dir + Utils::getFileNameWOExt(path) + "/";
    }

    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["mapNodeOm"].empty())
    {
        fs["mapNodeOm"] >> mapNodeOm;
    }

    std::map<int, std::vector<int>> bestCovisibleKeyFrameIdsMap;
    std::map<int, std::vector<int>> bestCovisibleWeightsMap;

    bool updateKeyFrameConnections = false;

    cv::FileNode n = fs["KeyFrames"];
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        int id = (*it)["id"];
        if (!(*it)["parentId"].empty())
        {
            int parentId    = (*it)["parentId"];
            parentIdMap[id] = parentId;
        }
        if (!(*it)["loopEdges"].empty() && (*it)["loopEdges"].isSeq())
        {
            cv::FileNode     les = (*it)["loopEdges"];
            std::vector<int> loopEdges;
            for (auto itLes = les.begin(); itLes != les.end(); ++itLes)
            {
                loopEdges.push_back((int)*itLes);
            }
            loopEdgesMap[id] = loopEdges;
        }
        cv::Mat Tcw; // has to be here!
        (*it)["Tcw"] >> Tcw;

        cv::Mat featureDescriptors; // has to be here!
        (*it)["featureDescriptors"] >> featureDescriptors;
        std::vector<cv::KeyPoint> keyPtsUndist;
        (*it)["keyPtsUndist"] >> keyPtsUndist;

        std::vector<int>   wordsId;
        std::vector<float> tfIdf;
        if (!(*it)["BowVectorWordsId"].empty())
            (*it)["BowVectorWordsId"] >> wordsId;
        if (!(*it)["TfIdf"].empty())
            (*it)["TfIdf"] >> tfIdf;

        float scaleFactor;
        (*it)["scaleFactor"] >> scaleFactor;
        int nScaleLevels = -1;
        (*it)["nScaleLevels"] >> nScaleLevels;

        // vectors for precalculation of scalefactors
        std::vector<float> vScaleFactor;
        std::vector<float> vInvScaleFactor;
        std::vector<float> vLevelSigma2;
        std::vector<float> vInvLevelSigma2;
        vScaleFactor.clear();
        vLevelSigma2.clear();
        vScaleFactor.resize(nScaleLevels);
        vLevelSigma2.resize(nScaleLevels);
        vScaleFactor[0] = 1.0f;
        vLevelSigma2[0] = 1.0f;
        for (int i = 1; i < nScaleLevels; i++)
        {
            vScaleFactor[i] = vScaleFactor[i - 1] * scaleFactor;
            vLevelSigma2[i] = vScaleFactor[i] * vScaleFactor[i];
        }

        vInvScaleFactor.resize(nScaleLevels);
        vInvLevelSigma2.resize(nScaleLevels);
        for (int i = 0; i < nScaleLevels; i++)
        {
            vInvScaleFactor[i] = 1.0f / vScaleFactor[i];
            vInvLevelSigma2[i] = 1.0f / vLevelSigma2[i];
        }

        cv::Mat K;
        (*it)["K"] >> K;
        float fx, fy, cx, cy;
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);

        // image bounds
        float nMinX, nMinY, nMaxX, nMaxY;
        (*it)["nMinX"] >> nMinX;
        (*it)["nMinY"] >> nMinY;
        (*it)["nMaxX"] >> nMaxX;
        (*it)["nMaxY"] >> nMaxY;

        WAIKeyFrame* newKf = new WAIKeyFrame(Tcw,
                                             id,
                                             fixKfsAndMPts,
                                             fx,
                                             fy,
                                             cx,
                                             cy,
                                             keyPtsUndist.size(),
                                             keyPtsUndist,
                                             featureDescriptors,
                                             voc,
                                             nScaleLevels,
                                             scaleFactor,
                                             vScaleFactor,
                                             vLevelSigma2,
                                             vInvLevelSigma2,
                                             (int)nMinX,
                                             (int)nMinY,
                                             (int)nMaxX,
                                             (int)nMaxY,
                                             K);

        if (!wordsId.empty() && !tfIdf.empty())
        {
            WAIBowVector bow(wordsId, tfIdf);
            newKf->SetBowVector(bow);
        }

        if (imgDir != "")
        {
            stringstream ss;
            ss << imgDir << "kf" << id << ".jpg";
            // newKf->imgGray = kfImg;
            if (Utils::fileExists(ss.str()))
            {
                newKf->setTexturePath(ss.str());
                cv::Mat imgColor = cv::imread(ss.str());
                cv::cvtColor(imgColor, newKf->imgGray, cv::COLOR_BGR2GRAY);
            }
        }
        keyFrames.push_back(newKf);
        kfsMap[newKf->mnId] = newKf;

        if (!(*it)["bestCovisibleKeyFrameIds"].empty() &&
            !(*it)["bestCovisibleWeights"].empty())
        {
            std::vector<int> bestCovisibleKeyFrameIds;
            (*it)["bestCovisibleKeyFrameIds"] >> bestCovisibleKeyFrameIds;
            std::vector<int> bestCovisibleWeights;
            (*it)["bestCovisibleWeights"] >> bestCovisibleWeights;

            bestCovisibleKeyFrameIdsMap[newKf->mnId] = bestCovisibleKeyFrameIds;
            bestCovisibleWeightsMap[newKf->mnId]     = bestCovisibleWeights;
        }
        else
        {
            updateKeyFrameConnections = true;
        }
    }

    // set parent keyframe pointers into keyframes
    for (WAIKeyFrame* kf : keyFrames)
    {
        if (kf->mnId != 0)
        {
            auto itParentId = parentIdMap.find(kf->mnId);
            if (itParentId != parentIdMap.end())
            {
                int  parentId   = itParentId->second;
                auto itParentKf = kfsMap.find(parentId);
                if (itParentKf != kfsMap.end())
                    kf->ChangeParent(itParentKf->second);
                else
                    cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist of keyframe " << kf->mnId << "! FAIL" << endl;
            }
            else
                cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist of keyframe " << kf->mnId << "! FAIL" << endl;
        }
    }

    int numberOfLoopClosings = 0;
    // set loop edge pointer into keyframes
    for (WAIKeyFrame* kf : keyFrames)
    {
        auto it = loopEdgesMap.find(kf->mnId);
        if (it != loopEdgesMap.end())
        {
            const auto& loopEdgeIds = it->second;
            for (int loopKfId : loopEdgeIds)
            {
                auto loopKfIt = kfsMap.find(loopKfId);
                if (loopKfIt != kfsMap.end())
                {
                    kf->AddLoopEdge(loopKfIt->second);
                    numberOfLoopClosings++;
                }
                else
                    cerr << "[WAIMapIO] loadKeyFrames: Loop keyframe id does not exist! FAIL" << endl;
            }
        }
    }
    numLoopClosings = numberOfLoopClosings / 2;

    n = fs["MapPoints"];
    if (n.type() != cv::FileNode::SEQ)
    {
        cerr << "strings is not a sequence! FAIL" << endl;
    }

    bool needMapPointUpdate = false;
    for (auto it = n.begin(); it != n.end(); ++it)
    {
        // newPt->id( (int)(*it)["id"]);
        int id = (int)(*it)["id"];

        cv::Mat mWorldPos; // has to be here!
        (*it)["mWorldPos"] >> mWorldPos;

        WAIMapPoint* newPt = new WAIMapPoint(id, mWorldPos, fixKfsAndMPts);
        vector<int>  observingKfIds;
        (*it)["observingKfIds"] >> observingKfIds;
        vector<int> corrKpIndices;
        (*it)["corrKpIndices"] >> corrKpIndices;

        // get reference keyframe id
        int  refKfId    = (int)(*it)["refKfId"];
        bool refKFFound = false;

        if (kfsMap.find(refKfId) != kfsMap.end())
        {
            newPt->refKf(kfsMap[refKfId]);
            refKFFound = true;
        }
        else
        {
            cout << "no reference keyframe found!" << endl;
            if (observingKfIds.size())
            {
                // we use the first of the observing keyframes
                int kfId = observingKfIds[0];
                if (kfsMap.find(kfId) != kfsMap.end())
                {
                    newPt->refKf(kfsMap[kfId]);
                    refKFFound = true;
                }
            }
        }

        if (!(*it)["mfMaxDistance"].empty() &&
            !(*it)["mfMinDistance"].empty() &&
            !(*it)["mNormalVector"].empty() &&
            !(*it)["mDescriptor"].empty())
        {
            newPt->SetMaxDistance((float)(*it)["mfMaxDistance"]);
            newPt->SetMinDistance((float)(*it)["mfMinDistance"]);
            cv::Mat normal, descriptor;
            (*it)["mNormalVector"] >> normal;
            (*it)["mDescriptor"] >> descriptor;
            newPt->SetNormal(normal);
            newPt->SetDescriptor(descriptor);
        }
        else
        {
            needMapPointUpdate = true;
        }

        if (refKFFound)
        {
            // find and add pointers of observing keyframes to map point
            for (int i = 0; i < observingKfIds.size(); ++i)
            {
                const int kfId = observingKfIds[i];
                if (kfsMap.find(kfId) != kfsMap.end())
                {
                    WAIKeyFrame* kf = kfsMap[kfId];
                    kf->AddMapPoint(newPt, corrKpIndices[i]);
                    newPt->AddObservation(kf, corrKpIndices[i]);
                }
            }
            mapPoints.push_back(newPt);
        }
        else
        {
            delete newPt;
        }
    }

    std::cout << "update the covisibility graph, when all keyframes and mappoints are loaded" << std::endl;
    // update the covisibility graph, when all keyframes and mappoints are loaded
    WAIKeyFrame* firstKF           = nullptr;
    bool         buildSpanningTree = false;
    for (WAIKeyFrame* kf : keyFrames)
    {
        if (updateKeyFrameConnections)
        {
            // Update links in the Covisibility Graph, do not build the spanning tree yet
            kf->FindAndUpdateConnections(false);
        }
        else
        {
            std::map<WAIKeyFrame*, int> keyFrameWeightMap;

            std::vector<int> bestCovisibleKeyFrameIds = bestCovisibleKeyFrameIdsMap[kf->mnId];
            std::vector<int> bestCovisibleWeights     = bestCovisibleWeightsMap[kf->mnId];

            for (int i = 0; i < bestCovisibleKeyFrameIds.size(); i++)
            {
                int          keyFrameId        = bestCovisibleKeyFrameIds[i];
                int          weight            = bestCovisibleWeights[i];
                WAIKeyFrame* covisibleKF       = kfsMap[keyFrameId];
                keyFrameWeightMap[covisibleKF] = weight;
            }

            kf->UpdateConnections(keyFrameWeightMap, false);
        }
        if (kf->mnId == 0)
        {
            firstKF = kf;
        }
        else if (kf->GetParent() == NULL)
        {
            buildSpanningTree = true;
        }
    }

    wai_assert(firstKF && "Could not find keyframe with id 0\n");

    // Build spanning tree if keyframes have no parents (legacy support)
    if (buildSpanningTree)
    {
        // QueueElem: <unconnected_kf, graph_kf, weight>
        using QueueElem = std::tuple<WAIKeyFrame*, WAIKeyFrame*, int>;
        auto cmpQueue   = [](const QueueElem& left, const QueueElem& right)
        { return (std::get<2>(left) < std::get<2>(right)); };
        auto cmpMap = [](const pair<WAIKeyFrame*, int>& left, const pair<WAIKeyFrame*, int>& right)
        { return left.second < right.second; };
        std::set<WAIKeyFrame*> graph;
        std::set<WAIKeyFrame*> unconKfs;
        for (auto& kf : keyFrames)
            unconKfs.insert(kf);

        // pick first kf
        graph.insert(firstKF);
        unconKfs.erase(firstKF);

        while (unconKfs.size())
        {
            std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmpQueue)> q(cmpQueue);
            // update queue with keyframes with neighbous in the graph
            for (auto& unconKf : unconKfs)
            {
                const std::map<WAIKeyFrame*, int>& weights = unconKf->GetConnectedKfWeights();
                for (auto& graphKf : graph)
                {
                    auto it = weights.find(graphKf);
                    if (it != weights.end())
                    {
                        QueueElem newElem = std::make_tuple(unconKf, it->first, it->second);
                        q.push(newElem);
                    }
                }
            }

            if (q.size() == 0)
            {
                // no connection: the remaining keyframes are unconnected
                Utils::log("WAIMapStorage", "Error in building spanning tree: There are %i unconnected keyframes!");
                break;
            }
            else
            {
                // extract keyframe with shortest connection
                QueueElem topElem = q.top();
                // remove it from unconKfs and add it to graph
                WAIKeyFrame* newGraphKf = std::get<0>(topElem);
                unconKfs.erase(newGraphKf);
                newGraphKf->ChangeParent(std::get<1>(topElem));
                // std::cout << "Added kf " << newGraphKf->mnId << " with parent " << std::get<1>(topElem)->mnId << std::endl;
                // update parent
                graph.insert(newGraphKf);
            }
        }
    }

    if (needMapPointUpdate)
    {
        PROFILE_SCOPE("Updating MapPoints");

        // compute resulting values for map points
        for (WAIMapPoint*& mp : mapPoints)
        {
            // mean viewing direction and depth
            mp->UpdateNormalAndDepth();
            mp->ComputeDistinctiveDescriptors();
        }
    }

    for (WAIKeyFrame* kf : keyFrames)
    {
        if (kf->mBowVec.data.empty())
        {
            std::cout << "kf->mBowVec.data empty" << std::endl;
            continue;
        }
        waiMap->AddKeyFrame(kf);
        waiMap->GetKeyFrameDB()->add(kf);

        // Add keyframe with id 0 to this vector. Otherwise RunGlobalBundleAdjustment in LoopClosing after loop was detected crashes.
        if (kf->mnId == 0)
        {
            waiMap->mvpKeyFrameOrigins.push_back(kf);
        }
    }

    for (WAIMapPoint* point : mapPoints)
    {
        waiMap->AddMapPoint(point);
    }

    waiMap->setNumLoopClosings(numLoopClosings);
    return true;
}

void WAIMapStorage::saveKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string> vidname, const std::string& dir, const std::string outputKFMatchingFile)
{
    if (!Utils::dirExists(dir))
        Utils::makeDir(dir);

    std::ofstream ofs;
    ofs.open(dir + "/" + outputKFMatchingFile, std::ofstream::out);

    ofs << to_string(vidname.size()) << "\n";

    for (int i = 0; i < vidname.size(); i++)
    {
        vidname[i] = Utils::getFileName(vidname[i]);
        ofs << vidname[i] << "\n";
    }

    for (int i = 0; i < keyFrameVideoMatching.size(); i++)
    {
        if (keyFrameVideoMatching[i] >= 0)
        {
            ofs << to_string(i) + " " + to_string(keyFrameVideoMatching[i]) << "\n";
        }
    }
    ofs.close();
}

void WAIMapStorage::loadKeyFrameVideoMatching(std::vector<int>& keyFrameVideoMatching, std::vector<std::string>& vidname, const std::string& dir, const std::string kFMatchingFile)
{
    std::ifstream ifs(dir + "/" + kFMatchingFile);
    keyFrameVideoMatching.resize(1000, -1);

    int nVid;
    ifs >> nVid;
    vidname.resize(nVid);

    for (int i = 0; i < nVid; i++)
    {
        ifs >> vidname[i];
        vidname[i] = Utils::getFileName(vidname[i]);
    }

    int kfId;
    int vid;
    while (ifs >> kfId >> vid)
    {
        if (kfId > keyFrameVideoMatching.size())
        {
            keyFrameVideoMatching.resize(keyFrameVideoMatching.size() * 2, -1);
        }
        keyFrameVideoMatching[kfId] = vid;
    }

    ifs.close();
}
