//
// Created by vwm1 on 27/09/2021.
//

#ifndef SLPROJECT_IDSPEAKCAPTURE_H
#define SLPROJECT_IDSPEAKCAPTURE_H

#include <cstdint>
#include <IDSPeakInterface.h>

class IDSPeakCapture
{

public:
    static IDSPeakCapture& instance()
    {
        static IDSPeakCapture instance;
        return instance;
    }

private:
    int      _width;
    int      _height;
    uint8_t* _dataBGR;
    uint8_t* _dataGray;

    bool running = false;

public:
    void start();
    void grab();
    void stop();

    int width() { return _width; }
    int height() { return _height; }
    uint8_t* dataBGR() { return _dataBGR; }
    uint8_t* dataGray() { return _dataGray; }

private:
    IDSPeakCapture();
    ~IDSPeakCapture();
};

#endif // SLPROJECT_IDSPEAKCAPTURE_H
