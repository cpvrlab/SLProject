//
// Created by vwm1 on 20/09/2021.
//

#ifndef SLPROJECT_SLGLFWINTERFACE_H
#define SLPROJECT_SLGLFWINTERFACE_H

#include <SL.h>
#include <SLEnums.h>
#include <GLFW/glfw3.h>

class SLGLFWInterface
{
private:
    static GLFWwindow* _window;

public:
    static void initialize();
    static GLFWwindow* createWindow(int width,
                             int height,
                             SLstring title,
                             int swapInterval,
                             int samples);
    static void createGLContext();

public:
    static SLKey mapKeyToSLKey(SLint key);

private:
    static void onGLFWError(int error, const char* description);

};

#endif // SLPROJECT_SLGLFWINTERFACE_H
