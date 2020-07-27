#ifndef DEVICE_DATA_H
#define DEVICE_DATA_H
#include <string>

//! Data that is different on different devices and different plattforms
class DeviceData
{
public:
    explicit DeviceData(int                scrWidth,
                        int                scrHeight,
                        int                dpi,
                        const std::string& dataDir,
                        const std::string& writableDir)
    {
        _dpi         = dpi;
        _scrWidth    = scrWidth;
        _scrHeight   = scrHeight;
        _dataDir     = Utils::unifySlashes(dataDir);
        _writableDir = Utils::unifySlashes(writableDir);
        //SLProject
        _shaderDir     = _dataDir + "shaders/";
        _vocabularyDir = _dataDir + "calibrations/";
        _fontDir       = _dataDir + "images/fonts/";
        //Erleb-AR productive
        _erlebARDir = _dataDir + "erleb-AR/";
        _textureDir = _dataDir + "erleb-AR/images/textures/";
        _stringsDir = _dataDir + "erleb-AR/strings/";
        //Erleb-AR test
        _erlebARTestDir      = _writableDir + "erleb-AR/";
        _erlebARCalibTestDir = _writableDir + "erleb-AR/calibrations/";
    }
    DeviceData() = delete;

    //screen width
    int scrWidth() const { return _scrWidth; }
    //screen height
    int scrHeight() const { return _scrHeight; }
    //dots per inch of screen
    int dpi() const { return _dpi; }
    //data directory (e.g. on desktop SLProject/data/. For mobile devices it is not visible to the user ("/data/user/0/ch.cpvr.wai/files"))
    const std::string& dataDir() const { return _dataDir; }
    //writable external app dir (e.g.AppData/Roaming on windows, <sd>/Android/data/ch.cpvr.wai/files on android)
    const std::string& writableDir() const { return _writableDir; }

    //path to fonts
    const std::string& fontDir() const { return _fontDir; }
    //path to textures
    const std::string& textureDir() const { return _textureDir; }
    //path to strings
    const std::string& stringsDir() const { return _stringsDir; }
    //path to shaders
    const std::string& shaderDir() const { return _shaderDir; }
    //path to vocabulary
    const std::string& vocabularyDir() const { return _vocabularyDir; }
    //internal erlebar dir (not visible to user on mobile devices)
    const std::string& erlebARDir() const { return _erlebARDir; }
    //directory containing calibrations for all our test devices
    const std::string& erlebARCalibTestDir() const { return _erlebARCalibTestDir; }
    //directory containing calibrations for all our test devices
    const std::string& erlebARTestDir() const { return _erlebARTestDir; }

private:
    //screen width
    int _scrWidth;
    //screen height
    int _scrHeight;
    //dots per inch of screen
    int _dpi;
    //data directory (e.g. on desktop SLProject/data/. For mobile devices it is not visible to the user)
    std::string _dataDir;
    //writable external app dir (e.g.AppData/Roaming on windows, <sd>/Android/data/ch.cpvr.wai/files on android)
    std::string _writableDir;

    //path to fonts
    std::string _fontDir;
    //path to textures
    std::string _textureDir;
    //path to strings in different languages
    std::string _stringsDir;
    //path to shaders
    std::string _shaderDir;
    //path to vocabulary
    std::string _vocabularyDir;
    //internal erlebar dir (not visible to user on mobile devices)
    std::string _erlebARDir;
    //external erlebar dir (for Testing, because we can access it)
    std::string _erlebARTestDir;
    //directory containing calibrations for all our test devices
    std::string _erlebARCalibTestDir;
};

#endif //DEVICE_DATA_H
