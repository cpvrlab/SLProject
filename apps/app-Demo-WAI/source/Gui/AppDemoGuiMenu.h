#ifndef APP_DEMOGUI_MENU_H
#define APP_DEMOGUI_MENU_H

#include <AppDemoGuiPrefs.h>
#include <SLSceneView.h>
#include <SLScene.h>

class AppDemoGuiMenu
{
    public:
    static void build(GUIPreferences * prefs, SLScene* s, SLSceneView* sv);
};

#endif
