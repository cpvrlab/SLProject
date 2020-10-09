#ifndef SIM_HELPER_GUI_H
#define SIM_HELPER_GUI_H

#include <string>
#include <imgui.h>

class SENSSimHelper;

class SimHelperGui
{
public:
    SimHelperGui(ImFont* fontText, ImFont* fontHeading, const char* title, float screenH)
      : _fontText(fontText),
        _fontHeading(fontHeading),
        _title("Simulation##" + std::string(title)),
        _screenH(screenH)
    {
    }

    void render(SENSSimHelper* simHelper);

    void reset()
    {
        _selectedSimData.clear();
    }
private:
    ImFont*        _fontText    = nullptr;
    ImFont*        _fontHeading = nullptr;
    std::string    _title;
    float          _screenH = 0.f;

    std::string _selectedSimData;
};

#endif
