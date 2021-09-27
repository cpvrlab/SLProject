//
// Created by vwm1 on 23/09/2021.
//

#ifndef SLPROJECT_IDSPEAKINTERFACE_H
#define SLPROJECT_IDSPEAKINTERFACE_H

#include <peak/peak.hpp>
#include <peak_ipl/peak_ipl.hpp>
#include <peak/converters/peak_buffer_converter_ipl.hpp>

class IDSPeakInterface
{

private:
    static std::shared_ptr<peak::core::Device>     device;
    static std::shared_ptr<peak::core::DataStream> dataStream;

    static std::shared_ptr<peak::core::NodeMap> nodeMapRemoteDevice;

public:
    static void init();
    static void openDevice();
    static void setDeviceParameters();
    static void allocateBuffers();

    static void startCapture();
    static void captureImage(int* width, int* height, uint8_t** dataBGR, uint8_t** dataGray);
    static void stopCapture();

    static void deallocateBuffers();
    static void uninit();
};

#endif // SLPROJECT_IDSPEAKINTERFACE_H
