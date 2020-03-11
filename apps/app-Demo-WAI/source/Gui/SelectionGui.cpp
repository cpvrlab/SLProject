#include <SelectionGui.h>
#include <SLScene.h>
#include <SLSceneView.h>

SelectionGui::SelectionGui(int dotsPerInch, std::string fontPath)
{
    // Scale for proportional and fixed size fonts
    float dpiScaleProp  = dotsPerInch / 120.0f;
    float dpiScaleFixed = dotsPerInch / 142.0f;

    // Default settings for the first time
    float fontPropDots  = std::max(16.0f * dpiScaleProp, 16.0f);
    float fontFixedDots = std::max(13.0f * dpiScaleFixed, 13.0f);

    //load fonts
    loadFonts(fontPropDots, fontFixedDots, fontPath);
}

void SelectionGui::build(SLScene* s, SLSceneView* sv)
{
    //draw buttons
    int i = 0;
}
