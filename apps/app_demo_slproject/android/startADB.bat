cd %ANDROID_HOME%\platform-tools\
adb start-server
adb devices
adb logcat -c
adb logcat SLProject *:S