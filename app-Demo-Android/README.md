# Build SLProject for Android

* [Download Android Studio for your platform](https://developer.android.com/studio/index.html)
* Install with all options left as default.
  * Remember where the Android SDK gets installed (e.g. c:\Users\???\AppData\Local\Android\sdk\)
* Startup **Android Studio** and choose **Import project (Eclipse ADT, Gradle, etc.)**
  * Choose the **build.gradle** file in the folder **app-Demo-Android**
* The SLProject demo app targets the **Android SDK 24** (= Android 6.0)
  * If not installed the IDE gives you a link for all missing SDK versions. 
  * The min. SDK needed is 21 (Android 5.0)
* The SLProject uses the **Android NDK** (native development kit). If not installed:
  * Open the **SDK manager** with **Tools > Android > SDK Manager**
  * Select the tab **SDK Tools**
    * Check the **NDK** option and click **Apply**
    * Check also the **Google USB Driver** option and click **Apply**
  * If the missing NDK error is still present, restart Android Studio and install possibly new package updates.
* The SLProject C++ library is configured with **cmake** in the **CMakeList.txt** file.
  * Install **cmake** if the IDE states it as missing.
* Build the project with **Hammer button** or with **Build > Make Project (Ctrl-F9)**
  * The app can only run on real but not on virtual devices because the prebuilt OpenCV libraries (_lib/prebuilt/Android) are only built for the arm architectures armeabi-v7a and arm64-v8a. To run the app in a virtual device you would have to build OpenCV also for Android on the x86 architecture.
* After successful build you can start the app on a USB-connected device by clicking the **green start button**.
  * If your device is not listed in the **Deployment Targets** dialog please check again your USB driver. In some cases, you have to download the appropriate driver from the device manufacturer.

