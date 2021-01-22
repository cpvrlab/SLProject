#ifndef DOWNLOAD_GUI_H
#define DOWNLOAD_GUI_H

#include <string>

#include <ImGuiWrapper.h>
#include <sm/EventSender.h>
#include <ErlebAR.h>
#include <Resources.h>
#include <AsyncWorker.h>

class SLScene;
class SLSceneView;
struct ImFont;

class DownloadGui : public ImGuiWrapper
  , private sm::EventSender
{
public:
    DownloadGui(const ImGuiEngine&                   imGuiEngine,
                sm::EventHandler&                    eventHandler,
                ErlebAR::Config&                     config,
                std::string                          dataDir,
                std::map<std::string, AsyncWorker*>& asyncWorkers,
                int                                  dotsPerInch,
                int                                  screenWidthPix,
                int                                  screenHeightPix);
    ~DownloadGui() override;

    void build(SLScene* s, SLSceneView* sv) override;
    void onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY) override;
    void onShow(); //call when gui becomes visible
    void initLocation(ErlebAR::LocationId locId) { _locId = locId; };

private:
    void download(std::string url, std::string dirname);
    void resize(int scrW, int scrH);

    ErlebAR::LocationId _locId;
    std::string         _dataDir;

    float _screenW;
    float _screenH;
    float _headerBarH;
    float _contentH;
    float _contentStartY;
    float _spacingBackButtonToText;
    float _buttonRounding;
    float _textWrapW;
    float _windowPaddingContent;
    float _itemSpacingContent;

    bool _download;

    ErlebAR::Config&    _config;
    ErlebAR::Resources& _resources;

    std::map<std::string, AsyncWorker*>& _asyncWorkers;
};

#endif //ABOUT_GUI_H
