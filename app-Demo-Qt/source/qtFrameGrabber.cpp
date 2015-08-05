#include "qtFrameGrabber.h"
#include <SLInterface.h>

//-----------------------------------------------------------------------------
qtFrameGrabber::qtFrameGrabber(QObject *parent) :
    QAbstractVideoSurface(parent)
{
    _camera.setViewfinder(this);
    lastFrameIsConsumed = true;
}
//-----------------------------------------------------------------------------
QList<QVideoFrame::PixelFormat> qtFrameGrabber::supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const
{
    Q_UNUSED(handleType);
    return QList<QVideoFrame::PixelFormat>()
        << QVideoFrame::Format_ARGB32
        << QVideoFrame::Format_ARGB32_Premultiplied
        << QVideoFrame::Format_RGB32
        << QVideoFrame::Format_RGB24
        << QVideoFrame::Format_RGB565
        << QVideoFrame::Format_RGB555
        << QVideoFrame::Format_ARGB8565_Premultiplied
        << QVideoFrame::Format_BGRA32
        << QVideoFrame::Format_BGRA32_Premultiplied
        << QVideoFrame::Format_BGR32
        << QVideoFrame::Format_BGR24
        << QVideoFrame::Format_BGR565
        << QVideoFrame::Format_BGR555
        << QVideoFrame::Format_BGRA5658_Premultiplied
        << QVideoFrame::Format_AYUV444
        << QVideoFrame::Format_AYUV444_Premultiplied
        << QVideoFrame::Format_YUV444
        << QVideoFrame::Format_YUV420P
        << QVideoFrame::Format_YV12
        << QVideoFrame::Format_UYVY
        << QVideoFrame::Format_YUYV
        << QVideoFrame::Format_NV12
        << QVideoFrame::Format_NV21
        << QVideoFrame::Format_IMC1
        << QVideoFrame::Format_IMC2
        << QVideoFrame::Format_IMC3
        << QVideoFrame::Format_IMC4
        << QVideoFrame::Format_Y8
        << QVideoFrame::Format_Y16
        << QVideoFrame::Format_Jpeg
        << QVideoFrame::Format_CameraRaw
        << QVideoFrame::Format_AdobeDng;
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::start()
{
    if (_camera.ActiveState != QCamera::ActiveState)
        _camera.start();
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::stop()
{
    if (_camera.ActiveState != QCamera::LoadedState)
        _camera.stop();
}
//-----------------------------------------------------------------------------
bool qtFrameGrabber::present(const QVideoFrame &frame)
{
    if (slNeedsVideoImage() && frame.isValid() && lastFrameIsConsumed)
    {
        QVideoFrame cloneFrame(frame);
        cloneFrame.map(QAbstractVideoBuffer::ReadOnly);

//        _lastFrame = QImage(cloneFrame.bits(),
//                            cloneFrame.width(),
//                            cloneFrame.height(),
//                            QVideoFrame::imageFormatFromPixelFormat(cloneFrame.pixelFormat()));
//        emit frameAvailable(image);

        // Set the according OpenGL format
        SLint glFormat;
        switch(cloneFrame.pixelFormat())
        {   case QVideoFrame::Format_Y8: glFormat = GL_LUMINANCE; break;
            case QVideoFrame::Format_BGR24: glFormat = GL_BGR; break;
            case QVideoFrame::Format_RGB24: glFormat = GL_RGB; break;
            case QVideoFrame::Format_ARGB32: glFormat = GL_RGBA; break;
            case QVideoFrame::Format_BGRA32: glFormat = GL_BGRA; break;
            default:
                SL_EXIT_MSG("qtCameraFrameGrabber::present: Qt image format not supported");
                cloneFrame.unmap();
                return false;
        }

        slCopyVideoImage(cloneFrame.width(), cloneFrame.height(), glFormat, cloneFrame.bits(), true);

        cloneFrame.unmap();

        lastFrameIsConsumed = false;
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
