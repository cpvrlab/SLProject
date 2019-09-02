//#############################################################################
//  File:      AppDemoGuiMapStorage.cpp
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <imgui.h>
#include <imgui_internal.h>

#include <Utils.h>
#include <AppDemoGuiMapStorage.h>

//-----------------------------------------------------------------------------
AppDemoGuiMapStorage::AppDemoGuiMapStorage(const string&      name,
                                           WAI::ModeOrbSlam2* tracking,
                                           SLNode*            mapNode,
                                           std::string        mapDir,
                                           bool*              activator)
  : AppDemoGuiInfosDialog(name, activator),
    _tracking(tracking),
    _mapNode(mapNode),
    _mapPrefix("slam-map-"),
    _nextId(0)
{
    wai_assert(tracking);
    _map  = tracking->getMap();
    _kfDB = tracking->getKfDB();

    _mapDir = Utils::unifySlashes(mapDir);

    _existingMapNames.clear();
    vector<pair<int, string>> existingMapNamesSorted;

    //check if visual odometry maps directory exists
    if (!Utils::dirExists(_mapDir))
    {
        Utils::makeDir(_mapDir);
    }
    else
    {
        //parse content: we search for directories in mapsDir
        std::vector<std::string> content = Utils::getFileNamesInDir(_mapDir);
        for (auto path : content)
        {
            std::string name = Utils::getFileName(path);
            //find json files that contain mapPrefix and estimate highest used id
            if (Utils::containsString(name, _mapPrefix))
            {
                //estimate highest used id
                std::vector<std::string> splitted;
                Utils::splitString(name, '-', splitted);
                if (splitted.size())
                {
                    int id = atoi(splitted.back().c_str());
                    existingMapNamesSorted.push_back(make_pair(id, name));
                    if (id >= _nextId)
                    {
                        _nextId = id + 1;
                    }
                }
            }
        }
    }

    //sort existingMapNames
    std::sort(existingMapNamesSorted.begin(), existingMapNamesSorted.end(), [](const pair<int, string>& left, const pair<int, string>& right) { return left.first < right.first; });
    for (auto it = existingMapNamesSorted.begin(); it != existingMapNamesSorted.end(); ++it)
        _existingMapNames.push_back(it->second);
}
//-----------------------------------------------------------------------------
void AppDemoGuiMapStorage::saveMap()
{
    //save keyframes (without graph/neigbourhood information)
    std::vector<WAIKeyFrame*> kfs = _tracking->getKeyFrames();
    if (kfs.size())
    {
        std::string mapDir   = _mapDir + _mapPrefix + std::to_string(_nextId) + "/";
        std::string filename = mapDir + _mapPrefix + std::to_string(_nextId) + ".json";
        std::string imgDir   = mapDir + "imgs";

        if (!Utils::dirExists(mapDir))
        {
            Utils::makeDir(mapDir);
        }
        else
        {
            if (Utils::fileExists(filename))
            {
                Utils::deleteFile(filename);
            }
        }

        if (!Utils::dirExists(imgDir))
        {
            Utils::makeDir(imgDir);
        }
        else
        {
            std::vector<std::string> content = Utils::getFileNamesInDir(imgDir);
            for (std::string path : content)
            {
                Utils::deleteFile(path);
            }
        }

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);

        SLMat4f om           = _mapNode->om();
        cv::Mat cvOm         = cv::Mat(4, 4, CV_32F);
        cvOm.at<float>(0, 0) = om.m(0);
        cvOm.at<float>(0, 1) = om.m(1);
        cvOm.at<float>(0, 2) = om.m(2);
        cvOm.at<float>(0, 3) = om.m(12);
        cvOm.at<float>(1, 0) = om.m(4);
        cvOm.at<float>(1, 1) = om.m(5);
        cvOm.at<float>(1, 2) = om.m(6);
        cvOm.at<float>(1, 3) = om.m(13);
        cvOm.at<float>(2, 0) = om.m(8);
        cvOm.at<float>(2, 1) = om.m(9);
        cvOm.at<float>(2, 2) = om.m(10);
        cvOm.at<float>(2, 3) = om.m(14);
        cvOm.at<float>(3, 0) = 0.f;
        cvOm.at<float>(3, 1) = 0.f;
        cvOm.at<float>(3, 2) = 0.f;
        cvOm.at<float>(3, 3) = 1.0f;
        fs << "mapNodeOm" << cvOm;

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
            if (_saveAndLoadImages)
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

        std::vector<WAIMapPoint*> mpts = _tracking->getMapPoints();
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
            //if((kfs.find(pKF) != mspKeyFrames.end()))
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

        _nextId++;

        ImGui::Text("Info: Map saved successfully");
    }
}
//-----------------------------------------------------------------------------
void AppDemoGuiMapStorage::buildInfos(SLScene* s, SLSceneView* sv)
{
    if (!_map)
        return;

    ImGui::Begin("Map storage", _activator, ImGuiWindowFlags_AlwaysAutoResize);
    if (ImGui::Button("Save map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        saveMap();
    }

    ImGui::Separator();
    if (ImGui::Button("New map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        //increase current id and maximum id in MapStorage
        WAIMapStorage::newMap();
        //clear current field in combobox, until this new map is saved
        WAIMapStorage::currItem = nullptr;
        WAIMapStorage::currN    = -1;
    }

    ImGui::Separator();
    {
        if (ImGui::BeginCombo("Current", WAIMapStorage::currItem)) // The second parameter is the label previewed before opening the combo.
        {
            for (int n = 0; n < _existingMapNames.size(); n++)
            {
                bool isSelected = (WAIMapStorage::currItem == _existingMapNames[n].c_str()); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(_existingMapNames[n].c_str(), isSelected))
                {
                    WAIMapStorage::currItem = _existingMapNames[n].c_str();
                    WAIMapStorage::currN    = n;
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
            }
            ImGui::EndCombo();
        }
    }

    if (ImGui::Button("Load map", ImVec2(ImGui::GetContentRegionAvailWidth(), 0.0f)))
    {
        if (WAIMapStorage::currItem)
        {
            //load selected map
            cv::Mat     cvOm            = cv::Mat(4, 4, CV_32F);
            std::string selectedMapName = _existingMapNames[WAIMapStorage::currN];

            _tracking->requestStateIdle();
            while (!_tracking->hasStateIdle())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            _tracking->reset();

            int selectedMapId = -1;

            //extract id from map name
            size_t prefixIndex = selectedMapName.find(_mapPrefix);
            if (prefixIndex != string::npos)
            {
                std::string name     = selectedMapName.substr(prefixIndex);
                std::string idString = name.substr(_mapPrefix.length());
                selectedMapId        = atoi(idString.c_str());

                std::string mapName  = _mapPrefix + idString;
                std::string mapDir   = _mapDir + mapName + "/";
                std::string filename = mapDir + mapName + ".json";
                std::string imgDir   = mapDir + "imgs";

                //check if dir and file exist
                if (Utils::dirExists(mapDir))
                {
                    if (Utils::fileExists(filename))
                    {
                        try
                        {
                            cv::FileStorage fs(filename, cv::FileStorage::READ);

                            fs["mapNodeOm"] >> cvOm;
                            SLMat4f om;
                            om.setMatrix(cvOm.at<float>(0, 0),
                                         cvOm.at<float>(0, 1),
                                         cvOm.at<float>(0, 2),
                                         cvOm.at<float>(0, 3),
                                         cvOm.at<float>(1, 0),
                                         cvOm.at<float>(1, 1),
                                         cvOm.at<float>(1, 2),
                                         cvOm.at<float>(1, 3),
                                         cvOm.at<float>(2, 0),
                                         cvOm.at<float>(2, 1),
                                         cvOm.at<float>(2, 2),
                                         cvOm.at<float>(2, 3),
                                         cvOm.at<float>(3, 0),
                                         cvOm.at<float>(3, 1),
                                         cvOm.at<float>(3, 2),
                                         cvOm.at<float>(3, 3));
                            _mapNode->om(om);

                            //mapping of keyframe pointer by their id (used during map points loading)
                            map<int, WAIKeyFrame*>    kfsMap;
                            std::vector<WAIKeyFrame*> keyFrames;
                            std::vector<WAIMapPoint*> mapPoints;
                            int                       numLoopClosings = 0;

                            {
                                cv::FileNode n = fs["KeyFrames"];
                                if (n.type() != cv::FileNode::SEQ)
                                {
                                    cerr << "strings is not a sequence! FAIL" << endl;
                                }

                                //the id of the parent is mapped to the kf id because we can assign it not before all keyframes are loaded
                                std::map<int, int> parentIdMap;
                                //vector of keyframe ids of connected loop edge candidates mapped to kf id that they are connected to
                                std::map<int, std::vector<int>> loopEdgesMap;
                                //reserve space in kfs
                                for (auto it = n.begin(); it != n.end(); ++it)
                                {
                                    int id = (*it)["id"];
                                    //load parent id
                                    if (!(*it)["parentId"].empty())
                                    {
                                        int parentId    = (*it)["parentId"];
                                        parentIdMap[id] = parentId;
                                    }
                                    //load ids of connected loop edge candidates
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
                                    // Infos about the pose: https://github.com/raulmur/ORB_SLAM2/issues/249
                                    // world w.r.t. camera pose -> wTc
                                    cv::Mat Tcw; //has to be here!
                                    (*it)["Tcw"] >> Tcw;

                                    cv::Mat featureDescriptors; //has to be here!
                                    (*it)["featureDescriptors"] >> featureDescriptors;

                                    //load undistorted keypoints in frame
                                    //todo: braucht man diese wirklich oder kann man das umgehen, indem zus�tzliche daten im MapPoint abgelegt werden (z.B. octave/level siehe UpdateNormalAndDepth)
                                    std::vector<cv::KeyPoint> keyPtsUndist;
                                    (*it)["keyPtsUndist"] >> keyPtsUndist;

                                    //ORB extractor information
                                    float scaleFactor;
                                    (*it)["scaleFactor"] >> scaleFactor;
                                    //number of pyriamid scale levels
                                    int nScaleLevels = -1;
                                    (*it)["nScaleLevels"] >> nScaleLevels;
                                    //calculation of scaleFactors , levelsigma2, invScaleFactors and invLevelSigma2
                                    //(copied from ORBextractor ctor)

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

                                    //calibration information
                                    //load camera matrix
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

                                    WAIKeyFrame* newKf = new WAIKeyFrame(Tcw, id, fx, fy, cx, cy, keyPtsUndist.size(), keyPtsUndist, featureDescriptors, WAIOrbVocabulary::get(), nScaleLevels, scaleFactor, vScaleFactor, vLevelSigma2, vInvLevelSigma2, nMinX, nMinY, nMaxX, nMaxY, K, _kfDB, _map);

                                    if (_saveAndLoadImages)
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

#if 1
                                    keyFrames.push_back(newKf);

                                    //pointer goes out of scope und wird invalid!!!!!!
                                    //map pointer by id for look-up
                                    kfsMap[newKf->mnId] = newKf;
#else
                                    //kfs.push_back(newKf);
                                    _map->AddKeyFrame(newKf);

                                    //Update keyframe database:
                                    //add to keyframe database
                                    _kfDB->add(newKf);

#endif
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
                                //there is a loop edge in the keyframe and the matched keyframe -> division by 2
                                numLoopClosings = numberOfLoopClosings / 2;
                            }

                            {
                                cv::FileNode n = fs["MapPoints"];
                                if (n.type() != cv::FileNode::SEQ)
                                {
                                    cerr << "strings is not a sequence! FAIL" << endl;
                                }

                                //reserve space in mapPts
                                //mapPts.reserve(n.size());
                                //read and add map points
                                for (auto it = n.begin(); it != n.end(); ++it)
                                {
                                    //newPt->id( (int)(*it)["id"]);
                                    int id = (int)(*it)["id"];

                                    cv::Mat mWorldPos; //has to be here!
                                    (*it)["mWorldPos"] >> mWorldPos;

                                    WAIMapPoint* newPt = new WAIMapPoint(id, mWorldPos, _map);
                                    //get observing keyframes
                                    vector<int> observingKfIds;
                                    (*it)["observingKfIds"] >> observingKfIds;
                                    //get corresponding keypoint indices in observing keyframe
                                    vector<int> corrKpIndices;
                                    (*it)["corrKpIndices"] >> corrKpIndices;

                                    //get reference keyframe id
                                    int  refKfId    = (int)(*it)["refKfId"];
                                    bool refKFFound = false;

                                    //check that reference keyframe exists (only use mappoints that have a valid reference keyframe)
                                    //If no reference keyframe exists, use the first observing keyframe. If there are no observers delete point.
                                    //TODO(michi) make sure every valid map point has a valid keyframe during SLAM
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
#if 1
                                        mapPoints.push_back(newPt);
#else
                                        _map->AddMapPoint(newPt);
#endif
                                    }
                                    else
                                    {
                                        delete newPt;
                                    }
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

                            _tracking->loadMapData(keyFrames, mapPoints, numLoopClosings);
                            _tracking->resume();

                            ImGui::Text("Slam map loading successful.");
                        }
                        catch (std::exception& e)
                        {
                            WAI_LOG("Exception while parsing map json file: %s", e.what());
                            ImGui::Text(("Failed to load map. " + filename).c_str());
                        }
                    }
                    else
                    {
                        ImGui::Text(("Failed to load map. " + filename).c_str());
                    }
                }
                else
                {
                    ImGui::Text(("Failed to load map. Path does not exist: " + mapDir).c_str());
                }
            }

#if 0
            if (WAIMapStorage::loadMap(selectedMapName, _tracking, WAIOrbVocabulary::get(), true, cvOm))
            {
                SLMat4f om;
                om.setMatrix(cvOm.at<float>(0, 0),
                             -cvOm.at<float>(0, 1),
                             -cvOm.at<float>(0, 2),
                             cvOm.at<float>(0, 3),
                             cvOm.at<float>(1, 0),
                             -cvOm.at<float>(1, 1),
                             -cvOm.at<float>(1, 2),
                             cvOm.at<float>(1, 3),
                             cvOm.at<float>(2, 0),
                             -cvOm.at<float>(2, 1),
                             -cvOm.at<float>(2, 2),
                             cvOm.at<float>(2, 3),
                             cvOm.at<float>(3, 0),
                             -cvOm.at<float>(3, 1),
                             -cvOm.at<float>(3, 2),
                             cvOm.at<float>(3, 3));
                _mapNode->om(om);
                ImGui::Text("Info: map loading successful!");
            }
            else
            {
                ImGui::Text("Info: map loading failed!");
            }
#endif
        }
    }
    ImGui::End();
}
