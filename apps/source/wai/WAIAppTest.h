#ifndef WAI_APP_TEST_H
#define WAI_APP_TEST_H

#include <string>
#include <SLInputEventInterface.h>
#include <SLGLTexture.h>
#include <SLCamera.h>

class SLSceneView;
class SENSCamera;

struct AppDirectories
{
    std::string writableDir;
    std::string waiDataRoot;
    std::string slDataRoot;
    std::string vocabularyDir;
    std::string logFileDir;
};

// implements app functionality (e.g. scene description, which camera, how to start and use WAISlam)
class WAIApp : public SLInputEventInterface
{
public:
    using CloseAppCallback = std::function<void()>;

    WAIApp();

    void init(int screenWidth, int screenHeight, int screenDpi, AppDirectories directories);
    void initCloseAppCallback(CloseAppCallback cb);
    void initCamera(SENSCamera* camera);

    bool update();
    void close();

    // back button pressed
    void goBack();

private:
    void initSceneCamera();
    void initDirectories(AppDirectories directories);

    // initialize SLScene, SLSceneView and UI
    void initSceneGraph(int scrWidth, int scrHeight, int dpi);
    void initIntroScene();
    void deleteSceneGraph();

    SENSCamera* _camera = nullptr;

    SLSceneView* _sv          = nullptr;
    SLCamera*    _sceneCamera = nullptr;
    SLGLTexture* _videoImage  = nullptr;

    AppDirectories _dirs;

    // implanted callback from WaiApp to system to
    CloseAppCallback _closeAppCallback;

    bool _goBackRequested    = false;
    bool _initSceneGraphDone = false;
    bool _initIntroSceneDone = false;
};

#endif // WAI_APP_TEST_H