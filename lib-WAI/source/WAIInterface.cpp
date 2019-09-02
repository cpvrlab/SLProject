#include <WAI.h>
#include <WAIMath.h>

static WAI::WAI           wai("");
static WAI::ModeOrbSlam2* mode = nullptr;

struct WAIMapPointCoordinate
{
    float x, y, z;
};

extern "C" {
WAI_API void wai_setDataRoot(const char* dataRoot)
{
    WAI_LOG("dataroot set to %s", dataRoot);
    wai.setDataRoot(dataRoot);
}

WAI_API void wai_setMode(WAI::ModeType modeType)
{
    WAI_LOG("setMode called");
    mode = (WAI::ModeOrbSlam2*)wai.setMode(modeType);
}

WAI_API void wai_getMapPoints(WAIMapPointCoordinate** mapPointCoordinatePtr,
                              int*                    mapPointCount)
{
    if (!mode)
    {
        WAI_LOG("mode not set. Call wai_setMode first.");
        return;
    }

    std::vector<WAIMapPoint*> mapPoints       = mode->getMapPoints();
    *mapPointCoordinatePtr                    = (WAIMapPointCoordinate*)malloc(mapPoints.size() * sizeof(WAIMapPointCoordinate));
    WAIMapPointCoordinate* mapPointCoordinate = *mapPointCoordinatePtr;

    int count = 0;

    for (WAIMapPoint* mapPoint : mapPoints)
    {
        if (!mapPoint->isBad())
        {
            *mapPointCoordinate = {
              mapPoint->worldPosVec().x,
              mapPoint->worldPosVec().y,
              mapPoint->worldPosVec().z,
            };

            mapPointCoordinate++;
            count++;
        }
    }

    *mapPointCount = count;
}

WAI_API void wai_releaseMapPoints(WAIMapPointCoordinate** mapPointCoordinatePtr)
{
    delete *mapPointCoordinatePtr;
}

WAI_API void wai_activateSensor(WAI::SensorType sensorType, void* sensorInfo)
{
    WAI_LOG("activateSensor called");
    wai.activateSensor(sensorType, sensorInfo);
}

WAI_API void wai_updateCamera(WAI::CameraFrame* frameRGB, WAI::CameraFrame* frameGray)
{
    WAI_LOG("updateCamera called");
    cv::Mat cvFrameRGB  = cv::Mat(frameRGB->height,
                                 frameRGB->width,
                                 CV_8UC3,
                                 frameRGB->memory,
                                 frameRGB->pitch);
    cv::Mat cvFrameGray = cv::Mat(frameGray->height,
                                  frameGray->width,
                                  CV_8UC1,
                                  frameGray->memory,
                                  frameGray->pitch);

    WAI::CameraData sensorData = {&cvFrameGray, &cvFrameRGB};

    wai.updateSensor(WAI::SensorType_Camera, &sensorData);
}

WAI_API bool wai_whereAmI(WAI::M4x4* pose)
{
    WAI_LOG("whereAmI called");
    bool result = 0;

    cv::Mat cvPose = cv::Mat(4, 4, CV_32F);
    result         = wai.whereAmI(&cvPose);

    if (result)
    {
        WAI_LOG("WAI knows where I am");
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                pose->e[x][y] = cvPose.at<float>(x, y);
            }
        }
    }

    return result;
}

WAI_API int wai_getState(char* buffer, int size)
{
    WAI_LOG("getState called");
    int result = 0;

    if (mode)
    {
        std::string state = mode->getPrintableState();

        if ((state.size() + 1) < size)
        {
            size = state.size() + 1;
        }

        result        = size;
        char*       c = buffer;
        const char* s = state.c_str();

        strncpy(c, s, size);
    }

    return result;
}

WAI_API void wai_registerDebugCallback(DebugLogCallback callback)
{
    registerDebugCallback(callback);
}
}
