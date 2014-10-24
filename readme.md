Welcome to the SLProject
========

SLProject is a plattform independent 3D computer graphics scene graph library


SL stands for Scene Library. It is developed at the [Berne University of Applied Sciences (BFH)](http://www.bfh.ch/en/studies/bachelor/engineering_and_information_technology/information_technology.html) and used for student projects in the [cpvrLab](https://www.cpvrlab.ti.bfh.ch/). The various applications show what you can learn in one semester about 3D computer graphics in real time rendering and ray tracing. During the semester we built this framework in C++ and OpenGL ES 2.0. The framework runs without changes on Windows, Linux, Mac OSX, Android (>=2.3.3) and Apple iOS (>=5). The framework can render alternatively with ray tracing which provides in addition high quality transparencies, reflections and soft shadows. You can find the demo app also on the [Android Market](https://play.google.com/store/apps/details?id=ch.fhnw.comgr&feature=search_result#?t=W251bGwsMSwyLDEsImNoLmZobncuY29tZ3IiXQ).


![alt text](http://slproject.googlecode.com/svn/images/platforms-small.png)

###Framework Features

 - **Platform independent real time 3D rendering** with OpenGL ES 2.0 on Windows, Mac OSX, Linux, Android and Apple iOS5.
 - **99% of the code is in platform independent C++** or **GLSL** (OpenGL Shading Language).
 - **Hierarchical scene graph** with grouping and referencing.
 - **3DS file format import**.
 - **Sorted alpha blending** for transparencies.
 - **View frustum culling** for optimal rendering of large scenes.
 - **Standard GLSL shaders** for vertex lighting, pixel lighting, bump mapping, parallax mapping, water and glass shading.
 - Turntable and first person camera animation.
 - **Stereo rendering** with anaglyphs, side-by-side or Oculus Rift rendering.
 - **Simple 2D GUI** buttons for hierarchical menus.
 - **Space partitioning** with axis aligned bounding volume hierarchy combined with uniform grids.
 - **Object picking** with fast ray casting.
 - Alternative rendering with **ray tracing**.
 - On desktop OS and iOS the ray tracing is optimally **paralleled using C++ 11 threads**.
 - **Distributed ray tracing** for anti aliasing, soft shadows and depth of field.
 - The project repository contains all dependencies and is **ready to compile**.
 - Read the wiki [BuildVisualStudio](https://code.google.com/p/slproject/wiki/BuildVisualStudio) for build instructions with MS Visual Studio.
 - Read the wiki [BuildQtCreator](https://code.google.com/p/slproject/wiki/BuildQtCreator) for build instructions with QtCreator? under Windows, OSX or Linux.
 - Read the wiki [BuildAndroid](https://code.google.com/p/slproject/wiki/BuildAndroid) for build instructions for the Android build with Visual Studio.
 - Read the wiki [BuildAppleiOS](https://code.google.com/p/slproject/wiki/BuildAppleiOS) for build instructions for the iOS build with Apple XCode.
 - Read the wiki FolderStructure for an overview of the repository folder structure.



###Minimal OpenGL ES2 Apps in 4 Languages
 - Minimal OpenGL application in **C++, C#, Java & WebGL**.
 - The projects can serve as entry points for core profile OpenGL programming newbies.
 - The apps show a simple rectangle with texture that can be rotated with the mouse implemented in:
 - C++ with the GLFW library for GUI.
 - C# with the OpenTK GL wrapper.
 - Java with the JOGL GL wrapper.
 - WebGL in JavaScript
 - The code in the 4 Languages is as identical as possible.



###Minimal 3D App without OpenGL
 - For learning 3D vector and matrix math you don't need OpenGL.
 - A spinning wireframe cube in 3D is created with 2D .Net drawing.
 - Is uses the same vector and matrix classes in C# as the SLProject uses in C++.
 - Perspective projection transform is implemented in the matrix class.




























