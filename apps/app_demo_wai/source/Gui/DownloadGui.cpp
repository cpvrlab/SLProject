#include <DownloadGui.h>
#include <imgui_internal.h>
#include <GuiUtils.h>
#include <ErlebAREvents.h>
#include <HttpUtils.h>
#include <ZipUtils.h>
#define PASSWORD "g@7bZ9bh5rkU"
using namespace ErlebAR;

class AsyncDownloader : public AsyncWorker
{
  public:
      float             _progress;
      int               _filesize;
      std::string       _dst;
      std::string       _url;
      ErlebAR::Location _loc;
      std::mutex        _mutex;
      bool              _hasStarted;

      AsyncDownloader(std::string url, std::string dst, ErlebAR::Location loc) : _url(url), _dst(dst), _loc(loc)
      {
          _filesize   = HttpUtils::length(url, "erlebar", PASSWORD);
          _progress   = 0.0f;
          _hasStarted = false;
    }

    int filesize()
    {
        return _filesize;
    }

    float progress()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        float p = _progress;
        lock.unlock();
        return p;
    }

    bool hasStarted()
    {
        return _hasStarted;
    }

    void setStarted()
    {
        _hasStarted = true;
    }

    void run()
    {
        HttpUtils::download(_url,
                            _dst,
                            "erlebar",
                            PASSWORD,
                            [this](size_t curr, size_t filesize) -> int
                            {
                                std::unique_lock<std::mutex> lock(_mutex);
                                _progress = (float)curr / (float)filesize;
                                lock.unlock();
                                return stopRequested();
                            });

        ZipUtils::unzip(_dst + _loc.dirName + ".zip", _dst);
        std::ofstream outfile (_dst + _loc.dirName + "/unzip_complete.txt");
        outfile.close();
        setReady();
    }
};



DownloadGui::DownloadGui(const ImGuiEngine&                   imGuiEngine,
                         sm::EventHandler&                    eventHandler,
                         ErlebAR::Config&                     config,
                         std::string                          dataDir,
                         std::map<std::string, AsyncWorker*>& asyncWorkers,
                         int                                  dotsPerInch,
                         int                                  screenWidthPix,
                         int                                  screenHeightPix)
  : ImGuiWrapper(imGuiEngine.context(), imGuiEngine.renderer()),
    sm::EventSender(eventHandler),
    _config(config),
    _dataDir(dataDir),
    _resources(config.resources()),
    _asyncWorkers(asyncWorkers)
{
    resize(screenWidthPix, screenHeightPix);
}

DownloadGui::~DownloadGui()
{
}

void DownloadGui::onShow()
{
    _panScroll.enable();
}

void DownloadGui::onResize(SLint scrW, SLint scrH, SLfloat scr2fbX, SLfloat scr2fbY)
{
    resize(scrW, scrH);
    ImGuiWrapper::onResize(scrW, scrH, scr2fbX, scr2fbY);
}

void DownloadGui::resize(int scrW, int scrH)
{
    _screenW = (float)scrW;
    _screenH = (float)scrH;

    _headerBarH              = _resources.style().headerBarPercH * _screenH;
    _contentH                = _screenH - _headerBarH;
    _contentStartY           = _headerBarH;
    _spacingBackButtonToText = _resources.style().headerBarSpacingBB2Text * _headerBarH;
    _buttonRounding          = _resources.style().buttonRounding * _screenH;
    _textWrapW               = 0.9f * _screenW;
    _windowPaddingContent    = _resources.style().windowPaddingContent * _screenH;
    _itemSpacingContent      = _resources.style().itemSpacingContent * _screenH;
}

void DownloadGui::build(SLScene* s, SLSceneView* sv)
{
    const auto& locations = _config.locations();
    auto        locIt     = locations.find(_locId);

    std::map<std::string, AsyncWorker*>::iterator asyncWorkerIt;
    AsyncDownloader* downloader = nullptr;

    if (locIt != locations.end())
    {
        ErlebAR::Location loc = locIt->second;
        if (Utils::fileExists(_dataDir + "erleb-AR/models/" + loc.dirName + "/unzip_complete.txt"))
        {
            sendEvent(new StartErlebarEvent("DownloadGui", _locId));
            return;
        }

        asyncWorkerIt = _asyncWorkers.find("download " + loc.dirName);
        if (asyncWorkerIt != _asyncWorkers.end())
        {
            downloader = (AsyncDownloader*)asyncWorkerIt->second;
            if (downloader->isReady())
            {
                downloader->stop();
                delete downloader;
                _asyncWorkers.erase(asyncWorkerIt);
            }
        }
        else
        {
            downloader = new AsyncDownloader(loc.url, _dataDir + "erleb-AR/models/", loc);
            _asyncWorkers.insert(std::pair<std::string, AsyncWorker*>("download " + loc.dirName, downloader));
        }
    }
    else
    {
        sendEvent(new GoBackEvent("DownloadGui"));
        return;
    }

    float buttonSize = _resources.style().headerBarButtonH * _headerBarH;

    ErlebAR::renderHeaderBar("DownloadGui",
                             _screenW,
                             _headerBarH,
                             _resources.style().headerBarBackgroundColor,
                             _resources.style().headerBarTextColor,
                             _resources.style().headerBarBackButtonColor,
                             _resources.style().headerBarBackButtonPressedColor,
                             _resources.fonts().headerBar,
                             _buttonRounding,
                             buttonSize,
                             _resources.textures.texIdBackArrow,
                             _spacingBackButtonToText,
                             _resources.strings().downloadManager(),
                             [&]() { sendEvent(new GoBackEvent("DownloadGui")); });

    //content
    {
        ImGui::SetNextWindowPos(ImVec2(0, _contentStartY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(_screenW, _contentH), ImGuiCond_Always);
        ImGuiWindowFlags childWindowFlags = ImGuiWindowFlags_NoTitleBar |
                                            ImGuiWindowFlags_NoMove |
                                            ImGuiWindowFlags_AlwaysAutoResize |
                                            ImGuiWindowFlags_NoBringToFrontOnFocus |
                                            ImGuiWindowFlags_NoScrollbar /*|
                                            ImGuiWindowFlags_NoScrollWithMouse*/
          ;
        ImGuiWindowFlags windowFlags = childWindowFlags |
                                       ImGuiWindowFlags_NoScrollWithMouse;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, _resources.style().backgroundColorPrimary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, _buttonRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(_windowPaddingContent, _windowPaddingContent));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(_windowPaddingContent, _windowPaddingContent));

        ImGui::Begin("DownloadGui_content", nullptr, windowFlags);
        ImGui::BeginChild("DownloadGui_content_child", ImVec2(0, 0), false, childWindowFlags);

        //general
        ImGui::PushFont(_resources.fonts().heading);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textHeadingColor);
        std::string str = std::string(_resources.strings().download1()) + std::to_string(downloader->filesize()/1000000) + " Mo" + std::string(_resources.strings().download2());
        ImGui::Text(str.c_str(), _textWrapW);
        ImGui::PopStyleColor();
        ImGui::PopFont();

        if (!downloader->hasStarted())
        {
            if (ImGui::Button(_resources.strings().downloadButton()))
            {
                downloader->start();
                downloader->setStarted();
            }
            else if (ImGui::Button(_resources.strings().downloadSkipButton()))
            {
                sendEvent(new StartErlebarEvent("DownloadGui", _locId));

            }
        }
        else
        {
            bool stop = ImGui::Button("Stop");
            ImGui::Separator();
            ImGui::ProgressBar(downloader->progress());
            ImGui::Separator();

            if (stop)
            {
                downloader->stop();
                delete downloader;
                _asyncWorkers.erase(asyncWorkerIt);
            }
        }

/*
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + _textWrapW);
        ImGui::PushFont(_resources.fonts().standard);
        ImGui::PushStyleColor(ImGuiCol_Text, _resources.style().textStandardColor);

        //Download other locations

        ImGui::PopStyleColor();
        ImGui::PopFont();
        ImGui::PopTextWrapPos();
*/

        ImGui::EndChild();
        ImGui::End();

        ImGui::PopStyleColor(1);
        ImGui::PopStyleVar(7);
    }

    //ImGui::ShowMetricsWindow();

    //debug: draw log window
    _config.logWinDraw();
}
