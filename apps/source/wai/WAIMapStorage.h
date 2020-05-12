#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <fbow.h>
#include <WAIModeOrbSlam2.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
public:
    static bool saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string fileName,
                        std::string imgDir = "");

    static bool loadMap(WAIMap*           waiMap,
                        SLNode*           mapNode,
                        fbow::Vocabulary* voc,
                        std::string       path,
                        bool              loadImgs,
                        bool              fixKfsAndMPts);

    static cv::Mat convertToCVMat(const SLMat4f slMat);
    static SLMat4f convertToSLMat(const cv::Mat& cvMat);
};

#endif
