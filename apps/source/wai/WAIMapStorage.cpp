#include <WAIMapStorage.h>

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

bool WAIMapStorage::saveMap(WAIMap*     waiMap,
                            SLNode*     mapNode,
                            std::string filename,
                            std::string imgDir)
{
    std::vector<WAIKeyFrame*> kfs  = waiMap->GetAllKeyFrames();
    std::vector<WAIMapPoint*> mpts = waiMap->GetAllMapPoints();

    //save keyframes (without graph/neigbourhood information)

    if (kfs.size())
    {
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

        //start sequence keyframes
        fs << "KeyFrames"
           << "[";
        for (int i = 0; i < kfs.size(); ++i)
        {
            WAIKeyFrame* kf = kfs[i];
            if (kf->isBad())
                continue;

            fs << "{"; //new map keyFrame
                       //add id
            fs << "id" << (int)kf->mnId;
            if (kf->mnId != 0) //kf with id 0 has no parent
                fs << "parentId" << (int)kf->GetParent()->mnId;
            else
                fs << "parentId" << -1;
            //loop edges: we store the id of the connected kf
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
            fs << "featureDescriptors" << kf->mDescriptors;
            fs << "keyPtsUndist" << kf->mvKeysUn;

            //scale factor
            fs << "scaleFactor" << kf->mfScaleFactor;
            //number of pyriamid scale levels
            fs << "nScaleLevels" << kf->mnScaleLevels;
            //fs << "fx" << kf->fx;
            //fs << "fy" << kf->fy;
            //fs << "cx" << kf->cx;
            //fs << "cy" << kf->cy;
            fs << "K" << kf->mK;

            //debug print
            //std::cout << "fx" << kf->fx << std::endl;
            //std::cout << "fy" << kf->fy << std::endl;
            //std::cout << "cx" << kf->cx << std::endl;
            //std::cout << "cy" << kf->cy << std::endl;
            //std::cout << "K" << kf->mK << std::endl;

            fs << "nMinX" << kf->mnMinX;
            fs << "nMinY" << kf->mnMinY;
            fs << "nMaxX" << kf->mnMaxX;
            fs << "nMaxY" << kf->mnMaxY;

            fs << "}"; //close map

            //save the original frame image for this keyframe
            if (imgDir != "")
            {
                cv::Mat imgColor;
                if (!kf->imgGray.empty())
                {
                    std::stringstream ss;
                    ss << imgDir << "kf" << (int)kf->mnId << ".jpg";

                    cv::cvtColor(kf->imgGray, imgColor, cv::COLOR_GRAY2BGR);
                    cv::imwrite(ss.str(), imgColor);

                    //if this kf was never loaded, we still have to set the texture path
                    kf->setTexturePath(ss.str());
                }
            }
        }
        fs << "]"; //close sequence keyframes

        //start map points sequence
        fs << "MapPoints"
           << "[";
        for (int i = 0; i < mpts.size(); ++i)
        {
            WAIMapPoint* mpt = mpts[i];
            //TODO: ghm1: check if it is necessary to removed points that have no reference keyframe OR can we somehow update the reference keyframe in the SLAM
            if (mpt->isBad() || mpt->refKf()->isBad())
                continue;

            fs << "{"; //new map for MapPoint
                       //add id
            fs << "id" << (int)mpt->mnId;
            //add position
            fs << "mWorldPos" << mpt->GetWorldPos();
            //save keyframe observations
            auto        observations = mpt->GetObservations();
            vector<int> observingKfIds;
            vector<int> corrKpIndices; //corresponding keypoint indices in observing keyframe
            for (auto it : observations)
            {
                if (!it.first->isBad())
                {
                    observingKfIds.push_back(it.first->mnId);
                    corrKpIndices.push_back(it.second);
                }
            }
            fs << "observingKfIds" << observingKfIds;
            fs << "corrKpIndices" << corrKpIndices;
            //(we calculate mean descriptor and mean deviation after loading)

            //reference key frame (I think this is the keyframe from which this
            //map point was generated -> first reference?)
            //if((kfs.find(pKF) != mspKeyFramstd::string(_nextId)es.end()))
            //if (!map.isKeyFrameInMap(mpt->refKf()))
            //{
            //    kfs.find(mpt->refKf())
            //    cout << "Reference keyframe not in map!" << endl;
            //}
            //else if (mpt->refKf()->isBad())
            //{
            //    cout << "Reference keyframe is bad!" << endl;
            //}
            fs << "refKfId" << (int)mpt->refKf()->mnId;
            fs << "}"; //close map
        }
        fs << "]";

        // explicit close
        fs.release();
    }
    else
    {
        return false;
    }
    return true;
}

bool WAIMapStorage::loadMap(WAIMap*        waiMap,
                            SLNode*        mapNode,
                            ORBVocabulary* voc,
                            std::string    path,
                            bool           loadImgs,
                            bool           fixKfsAndMPts)
{
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

    if (mapNode && !fs["mapNodeOm"].empty())
    {
        cv::Mat cvOm;
        fs["mapNodeOm"] >> cvOm;
        SLMat4f slOm = convertToSLMat(cvOm);
        std::cout << "slOm: " << slOm.toString() << std::endl;

        mapNode->om(slOm);
    }

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
        cv::Mat Tcw; //has to be here!
        (*it)["Tcw"] >> Tcw;

        cv::Mat featureDescriptors; //has to be here!
        (*it)["featureDescriptors"] >> featureDescriptors;
        std::vector<cv::KeyPoint> keyPtsUndist;
        (*it)["keyPtsUndist"] >> keyPtsUndist;
        float scaleFactor;
        (*it)["scaleFactor"] >> scaleFactor;
        int nScaleLevels = -1;
        (*it)["nScaleLevels"] >> nScaleLevels;

        //vectors for precalculation of scalefactors
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

        //image bounds
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
                                             nMinX,
                                             nMinY,
                                             nMaxX,
                                             nMaxY,
                                             K);

        if (imgDir != "")
        {
            stringstream ss;
            ss << imgDir << "kf" << id << ".jpg";
            //newKf->imgGray = kfImg;
            if (Utils::fileExists(ss.str()))
            {
                newKf->setTexturePath(ss.str());
                cv::Mat imgColor = cv::imread(ss.str());
                cv::cvtColor(imgColor, newKf->imgGray, cv::COLOR_BGR2GRAY);
            }
        }
        keyFrames.push_back(newKf);
        kfsMap[newKf->mnId] = newKf;
    }

    //set parent keyframe pointers into keyframes
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
                    cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist! FAIL" << endl;
            }
            else
                cerr << "[WAIMapIO] loadKeyFrames: Parent does not exist! FAIL" << endl;
        }
    }

    int numberOfLoopClosings = 0;
    //set loop edge pointer into keyframes
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

    for (auto it = n.begin(); it != n.end(); ++it)
    {
        //newPt->id( (int)(*it)["id"]);
        int id = (int)(*it)["id"];

        cv::Mat mWorldPos; //has to be here!
        (*it)["mWorldPos"] >> mWorldPos;

        WAIMapPoint* newPt = new WAIMapPoint(id, mWorldPos, fixKfsAndMPts);
        vector<int>  observingKfIds;
        (*it)["observingKfIds"] >> observingKfIds;
        vector<int> corrKpIndices;
        (*it)["corrKpIndices"] >> corrKpIndices;

        //get reference keyframe id
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
                //we use the first of the observing keyframes
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
            //find and add pointers of observing keyframes to map point
            for (int i = 0; i < observingKfIds.size(); ++i)
            {
                const int kfId = observingKfIds[i];
                if (kfsMap.find(kfId) != kfsMap.end())
                {
                    WAIKeyFrame* kf = kfsMap[kfId];
                    kf->AddMapPoint(newPt, corrKpIndices[i]);
                    newPt->AddObservation(kf, corrKpIndices[i]);
                }
                else
                {
                    cout << "keyframe with id " << i << " not found!";
                }
            }
            mapPoints.push_back(newPt);
        }
        else
        {
            delete newPt;
        }
    }

    //update the covisibility graph, when all keyframes and mappoints are loaded
    WAIKeyFrame* firstKF           = nullptr;
    bool         buildSpanningTree = false;
    for (WAIKeyFrame* kf : keyFrames)
    {
        // Update links in the Covisibility Graph, do not build the spanning tree yet
        kf->UpdateConnections(false);
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
        //QueueElem: <unconnected_kf, graph_kf, weight>
        using QueueElem                 = std::tuple<WAIKeyFrame*, WAIKeyFrame*, int>;
        auto                   cmpQueue = [](const QueueElem& left, const QueueElem& right) { return (std::get<2>(left) < std::get<2>(right)); };
        auto                   cmpMap   = [](const pair<WAIKeyFrame*, int>& left, const pair<WAIKeyFrame*, int>& right) { return left.second < right.second; };
        std::set<WAIKeyFrame*> graph;
        std::set<WAIKeyFrame*> unconKfs;
        for (auto& kf : keyFrames)
            unconKfs.insert(kf);

        //pick first kf
        graph.insert(firstKF);
        unconKfs.erase(firstKF);

        while (unconKfs.size())
        {
            std::priority_queue<QueueElem, std::vector<QueueElem>, decltype(cmpQueue)> q(cmpQueue);
            //update queue with keyframes with neighbous in the graph
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
            //extract keyframe with shortest connection
            QueueElem topElem = q.top();
            //remove it from unconKfs and add it to graph
            WAIKeyFrame* newGraphKf = std::get<0>(topElem);
            unconKfs.erase(newGraphKf);
            newGraphKf->ChangeParent(std::get<1>(topElem));
            std::cout << "Added kf " << newGraphKf->mnId << " with parent " << std::get<1>(topElem)->mnId << std::endl;
            //update parent
            graph.insert(newGraphKf);
        }
    }

    //compute resulting values for map points
    for (WAIMapPoint*& mp : mapPoints)
    {
        //mean viewing direction and depth
        mp->UpdateNormalAndDepth();
        mp->ComputeDistinctiveDescriptors();
    }

    for (WAIKeyFrame* kf : keyFrames)
    {
        waiMap->AddKeyFrame(kf);

        //Add keyframe with id 0 to this vector. Otherwise RunGlobalBundleAdjustment in LoopClosing after loop was detected crashes.
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
