#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <OrbSlam/ORBVocabulary.h>
#include <WAIModeOrbSlam2.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
    public:
    static bool saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string featureType,
                        std::string path,
                        std::string imgDir = "");

    static bool loadMap(WAIMap*        waiMap,
                        WAIKeyFrameDB* kfDB,
                        SLNode*        mapNode,
                        std::string    path,
                        std::string    imgDir = "");

    private:
    static SLMat4f loadMatrix(const cv::FileNode& n);
};

#endif
