//#############################################################################
//  File:      IDSPeakDevice.cpp
//  Date:      November 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marino von Wattenwyl
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <IDSPeakDevice.h>
#include <IDSPeakInterface.h>
#include <Instrumentor.h>

IDSPeakDevice::IDSPeakDevice()
{
}
//-----------------------------------------------------------------------------
IDSPeakDevice::IDSPeakDevice(std::shared_ptr<peak::core::Device> device,
                             int                                 deviceIndex)
  : _device(device), _deviceIndex(deviceIndex)
{
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::prepare(IDSPeakDeviceParams& params)
{
    setParameters(params);
    allocateBuffers();
    startCapture();
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::setParameters(IDSPeakDeviceParams& params)
{
    try
    {
        _nodeMapRemoteDevice = _device->RemoteDevice()->NodeMaps().at(0);

        // Set frame rate to maximum
        auto frameRateNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::FloatNode>("AcquisitionFrameRate");
        frameRateNode->SetValue(params.frameRate);
        SL_LOG("IDS Peak: Frame rate = %f Hz", frameRateNode->Value());

        // Set region of interest
        auto offsetXNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("OffsetX");
        auto offsetYNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("OffsetY");
        auto widthNode   = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("Width");
        auto heightNode  = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("Height");

        offsetXNode->SetValue(0);
        offsetYNode->SetValue(0);
        widthNode->SetValue(widthNode->Maximum());
        heightNode->SetValue(heightNode->Maximum());

        SL_LOG("IDS Peak: Region of Interest = [x: %d, y: %d, width: %d, height: %d]",
               offsetXNode->Value(),
               offsetYNode->Value(),
               widthNode->Value(),
               heightNode->Value());

        // Set gain
        _gainNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::FloatNode>("Gain");
        _gainNode->SetValue(params.gain);
        SL_LOG("IDS Peak: Gain = %f", _gainNode->Value());

        // Set gamma
        _gammaNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::FloatNode>("Gamma");
        _gammaNode->SetValue(params.gamma);
        SL_LOG("IDS Peak: Gamma = %f", _gammaNode->Value());

        // Set binning
        auto binningHNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("BinningHorizontal");
        auto binningVNode = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("BinningVertical");
        binningHNode->SetValue(params.binning);
        binningVNode->SetValue(params.binning);
    }
    catch (const std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::setParameters");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::allocateBuffers()
{
    try
    {
        auto dataStreams = _device->DataStreams();
        if (dataStreams.empty())
        {
            SL_EXIT_MSG("IDS Peak: Device has no data streams");
        }

        _dataStream                                            = dataStreams.at(0)->OpenDataStream();
        std::shared_ptr<peak::core::NodeMap> nodeMapDataStream = _dataStream->NodeMaps().at(0);

        // Get the payload size
        int64_t payloadSize = _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();
        SL_LOG("IDS Peak: Payload size = %d bytes", (int)payloadSize);

        // Get the number of required buffers
        int numBuffers = (int)_dataStream->NumBuffersAnnouncedMinRequired();

        // Allocate and queue the buffers
        for (int i = 0; i < numBuffers; i++)
        {
            auto buffer = _dataStream->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
            _dataStream->QueueBuffer(buffer);
        }

        SL_LOG("IDS Peak: Buffers allocated");
    }
    catch (const std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::allocateBuffers");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::startCapture()
{
    try
    {
        _dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default, PEAK_INFINITE_NUMBER);

        _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(1);
        _nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();

        SL_LOG("IDS Peak: Capture started");
    }
    catch (const std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::startCapture");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::captureImage(int*      width,
                                 int*      height,
                                 uint8_t** dataBGR,
                                 uint8_t** dataGray)
{
    PROFILE_FUNCTION();

    try
    {
        const auto buffer    = _dataStream->WaitForFinishedBuffer(5000);
        const auto image     = peak::ipl::Image(peak::BufferTo<peak::ipl::Image>(buffer));
        const auto imageBGR  = image.ConvertTo(peak::ipl::PixelFormatName::BGR8,
                                               peak::ipl::ConversionMode::Fast);
        const auto imageGray = image.ConvertTo(peak::ipl::PixelFormatName::Mono8,
                                               peak::ipl::ConversionMode::Fast);

        *width    = (int)image.Width();
        *height   = (int)image.Height();
        *dataBGR  = imageBGR.Data();
        *dataGray = imageGray.Data();

        _dataStream->QueueBuffer(buffer);
    }
    catch (std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::captureImage");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
double IDSPeakDevice::gain()
{
    return _gainNode->Value();
}
//-----------------------------------------------------------------------------
double IDSPeakDevice::gamma()
{
    return _gammaNode->Value();
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::gain(double gain)
{
    _gainNode->SetValue(gain);
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::gamma(double gamma)
{
    _gammaNode->SetValue(gamma);
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::close()
{
    stopCapture();
    deallocateBuffers();
    IDSPeakInterface::instance().deviceClosed();
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::stopCapture()
{
    try
    {
        _nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStop")->Execute();
        _nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(0);

        _dataStream->StopAcquisition(peak::core::AcquisitionStopMode::Default);

        SL_LOG("IDS Peak: Capture stopped");
    }
    catch (std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::stopCapture");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------
void IDSPeakDevice::deallocateBuffers()
{
    try
    {
        _dataStream->Flush(peak::core::DataStreamFlushMode::DiscardAll);

        for (const auto& buffer : _dataStream->AnnouncedBuffers())
        {
            _dataStream->RevokeBuffer(buffer);
        }
    }
    catch (std::exception& e)
    {
        SL_LOG("Exception in IDSPeakDevice::deallocateBuffers");
        SL_EXIT_MSG(e.what());
    }
}
//-----------------------------------------------------------------------------