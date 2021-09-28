#include "IDSPeakInterface.h"

#include <SL.h>

std::shared_ptr<peak::core::Device>     IDSPeakInterface::device              = nullptr;
std::shared_ptr<peak::core::DataStream> IDSPeakInterface::dataStream          = nullptr;
std::shared_ptr<peak::core::NodeMap>    IDSPeakInterface::nodeMapRemoteDevice = nullptr;

void IDSPeakInterface::init()
{
    try
    {
        peak::Library::Initialize();
        SL_LOG("IDS Peak: Initialized");
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::openDevice()
{
    try
    {
        auto& deviceManager = peak::DeviceManager::Instance();

        // Search for available devices
        deviceManager.Update();

        auto devices = deviceManager.Devices();
        SL_LOG("IDS Peak: Found %d available device(s)", devices.size());

        if (!devices.empty())
        {
            // Open the first available device
            device = devices.at(0)->OpenDevice(peak::core::DeviceAccessType::Control);
            SL_LOG("IDS Peak: Device \"%s\" opened", device->DisplayName().c_str());
        }
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::setDeviceParameters()
{
    try
    {
        nodeMapRemoteDevice = device->RemoteDevice()->NodeMaps().at(0);

        // Set frame rate to maximum
        auto frameRateNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::FloatNode>("AcquisitionFrameRate");
        frameRateNode->SetValue(frameRateNode->Maximum());
        SL_LOG("IDS Peak: Frame rate = %f Hz", frameRateNode->Value());

        // Set region of interest
        auto offsetXNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("OffsetX");
        auto offsetYNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("OffsetY");
        auto widthNode   = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("Width");
        auto heightNode  = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("Height");

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
        auto gainNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::FloatNode>("Gamma");
        gainNode->SetValue(2.5f);
        SL_LOG("IDS Peak: Gain = %f", gainNode->Value());

        // Set binning
        auto binningHNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("BinningHorizontal");
        auto binningVNode = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("BinningVertical");
        binningHNode->SetValue(2);
        binningVNode->SetValue(2);
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::allocateBuffers()
{
    try
    {
        auto dataStreams = device->DataStreams();
        if (dataStreams.empty())
        {
            SL_EXIT_MSG("IDS Peak: Device has no data streams");
        }

        dataStream                                             = dataStreams.at(0)->OpenDataStream();
        std::shared_ptr<peak::core::NodeMap> nodeMapDataStream = dataStream->NodeMaps().at(0);

        // Get the payload size
        int64_t payloadSize = nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("PayloadSize")->Value();
        SL_LOG("IDS Peak: Payload size = %d bytes", (int)payloadSize);

        // Get the number of required buffers
        int numBuffers = (int)dataStream->NumBuffersAnnouncedMinRequired();

        // Allocate the buffers
        for (int i = 0; i < numBuffers; i++)
        {
            auto buffer = dataStream->AllocAndAnnounceBuffer(static_cast<size_t>(payloadSize), nullptr);
            dataStream->QueueBuffer(buffer);
        }

        SL_LOG("IDS Peak: Buffers allocated");
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::startCapture()
{
    try
    {
        dataStream->StartAcquisition(peak::core::AcquisitionStartMode::Default, PEAK_INFINITE_NUMBER);

        nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(1);
        nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStart")->Execute();

        SL_LOG("IDS Peak: Capture started");
    }
    catch (const std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::captureImage(int* width, int* height, uint8_t** dataBGR, uint8_t** dataGray)
{
    try
    {
        const auto buffer    = dataStream->WaitForFinishedBuffer(500);
        const auto image     = peak::ipl::Image(peak::BufferTo<peak::ipl::Image>(buffer));
        const auto imageBGR  = image.ConvertTo(peak::ipl::PixelFormatName::BGR8,
                                               peak::ipl::ConversionMode::Fast);
        const auto imageGray = image.ConvertTo(peak::ipl::PixelFormatName::Mono8,
                                               peak::ipl::ConversionMode::Fast);

        *width    = (int)image.Width();
        *height   = (int)image.Height();
        *dataBGR  = imageBGR.Data();
        *dataGray = imageGray.Data();

        dataStream->QueueBuffer(buffer);
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::stopCapture()
{
    try
    {
        nodeMapRemoteDevice->FindNode<peak::core::nodes::CommandNode>("AcquisitionStop")->Execute();
        nodeMapRemoteDevice->FindNode<peak::core::nodes::IntegerNode>("TLParamsLocked")->SetValue(0);

        dataStream->StopAcquisition(peak::core::AcquisitionStopMode::Default);

        SL_LOG("IDS Peak: Capture stopped");
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::deallocateBuffers()
{
    try
    {
        dataStream->Flush(peak::core::DataStreamFlushMode::DiscardAll);

        for (const auto& buffer : dataStream->AnnouncedBuffers())
        {
            dataStream->RevokeBuffer(buffer);
        }
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}

void IDSPeakInterface::uninit()
{
    try
    {
        peak::Library::Close();
        SL_LOG("IDS Peak: Uninitialized");
    }
    catch (std::exception& e)
    {
        SL_EXIT_MSG(e.what());
    }
}