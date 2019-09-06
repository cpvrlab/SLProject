#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <SLSceneView.h>
#include <WAIHelper.h>
#include <OrbSlam/ORBVocabulary.h>
#include <WAIModeOrbSlam2.h>
#include <WAIMapIO.h>
#include <Utils.h>

class WAI_API WAIMapStorage
{
    public:
    static void saveMap(WAIMap*     waiMap,
                        SLNode*     mapNode,
                        std::string path,
                        std::string imgDir = "");

    static void loadMap(WAIMap*        waiMap,
                        WAIKeyFrameDB* kfDB,
                        SLNode*        mapNode,
                        std::string    path,
                        std::string    imgDir = "");

    private:
    static SLMat4f loadMatrix(const cv::FileNode& n);
};

#endif
