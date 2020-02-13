#ifndef WAI_APP_H
#define WAI_APP_H

#include <string>
#include <SL/SLInputEventInterface.h>

class SLSceneView;

struct AppDirectories
{
    std::string writableDir;
    std::string waiDataRoot;
    std::string slDataRoot;
    std::string vocabularyDir;
    std::string logFileDir;
};

//implements app functionality (e.g. scene description, which camera, how to start and use WAISlam)
class WAIApp
{
public:
    bool render();

    //---------------------------------------------------------
    //initialization methods:

    void initDirectories(AppDirectories directories);
    //initialize SLScene, SLSceneView and UI
    void initSceneGraph(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi);
    void initIntroScene();

    void deleteSceneGraphe();
    //camera start is initiated
    //void startCamera() {}
    //void stopCamera() {}

private:
    SLSceneView*   _sv = nullptr;
    AppDirectories _dirs;
};

//implements logig about which scenes to load when and what is needed for a state transition
class WAIAppStateHandler : public SLInputEventInterface
{
public:
    using CloseAppCallback = std::function<void()>;

    WAIAppStateHandler(CloseAppCallback cb);

    enum class State
    {
        STARTUP,
        INTROSCENE,
        START_SLAM_SCENE,
        SLAM_SCENE,
    };

    //returns true if screen needs an update
    bool update();

    //--------------------------------------------------------
    //Events

    //app is initially started after we have a context
    void init(int screenWidth, int screenHeight, float scr2fbX, float scr2fbY, int screenDpi, AppDirectories directories);
    //app is put in foreground
    void show();
    //app was put into background (stop camera, make sure all processing threads wait)
    void hide();
    //close app: if not hidden hide and free all storage
    void close();
    //back button was pressed
    void goBack();

private:
    void checkStateTransition();
    //returns true if screen needs an update
    bool processState();

    std::unique_ptr<WAIApp> _waiApp;
    State                   _state = State::STARTUP;

    //implanted callback from WaiApp to system to
    CloseAppCallback _closeAppCallback;

    bool _goBackRequested = false;

    bool _initSceneGraphDone   = false;
    bool _initIntroSceneDone = false;
};

#endif //WAI_APP_H