// #############################################################################
//   File:      WebCamera.h
//   Purpose:   Interface to access the camera through the browser.
//   Date:      October 2022
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marino von Wattenwyl
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
// #############################################################################

#include <WebCamera.h>
#include <emscripten.h>

//-----------------------------------------------------------------------------
void WebCamera::open(WebCameraFacing facing)
{
    // clang-format off
    MAIN_THREAD_EM_ASM({
        console.log("[WebCamera] Requesting stream...");

        let facingMode;
        if ($0 == 0) facingMode = "user";
        else if ($0 == 1) facingMode = "environment";

        navigator.mediaDevices.getUserMedia({ "video": { "facingMode": facingMode } })
            .then(stream => {
                console.log("[WebCamera] Stream acquired");

                let video = document.querySelector("#capture-video");
                video.srcObject = stream;
            })
            .catch(error => {
                console.log("[WebCamera] Failed to acquire stream");
                console.log(error);
            });
    }, facing);
    // clang-format on

    _isOpened = true;
}
//-----------------------------------------------------------------------------
bool WebCamera::isReady()
{
    return _image.cols != 0 && _image.rows != 0;
}
//-----------------------------------------------------------------------------
CVMat WebCamera::read()
{
    CVSize2i size = getSize();

    // If the width or the height is zero, the video is not ready
    if (size.width == 0 || size.height == 0)
        return CVMat(0, 0, CV_8UC3);

    // Recreate the image if the size has changed
    if (size.width != _image.cols || size.height != _image.rows)
    {
        _image = CVMat(size.height, size.width, CV_8UC4);
        _waitingForResize = false;
    }

    // clang-format off
    MAIN_THREAD_EM_ASM({
        let video = document.querySelector("#capture-video");
        let canvas = document.querySelector("#capture-canvas");
        let ctx = canvas.getContext("2d");

        let width = video.videoWidth;
        let height = video.videoHeight;

        if (width == 0 || height == 0)
            return;

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(video, 0, 0, width, height);
        let imageData = ctx.getImageData(0, 0, width, height);

        writeArrayToMemory(imageData.data, $0);
    }, _image.data);
    // clang-format on

    if (_imageBGR.size != _image.size)
        _imageBGR = CVMat(_image.rows, _image.cols, CV_8UC3);

    cv::cvtColor(_image, _imageBGR, cv::COLOR_RGBA2BGR);

    return _imageBGR;
}
//-----------------------------------------------------------------------------
CVSize2i WebCamera::getSize()
{
    int32_t width;
    int32_t height;

    // clang-format off
    MAIN_THREAD_EM_ASM({
        let video = document.querySelector("#capture-video");
        let width = video.videoWidth;
        let height = video.videoHeight;

        setValue($0, video.videoWidth, "i32");
        setValue($1, video.videoHeight, "i32");
    }, &width, &height);
    // clang-format on

    return CVSize2i(width, height);
}
//-----------------------------------------------------------------------------
void WebCamera::setSize(CVSize2i size)
{
    // Return if the stream is still loading
    if(!isReady())
        return;

    // Return if we are already waiting for the resize
    if(_waitingForResize)
        return;

    // Return if the new size is equal to the old size
    if (size.width == _image.cols && size.height == _image.rows)
        return;

    _waitingForResize = true;

    // clang-format off
    MAIN_THREAD_EM_ASM({
        let video = document.querySelector("#capture-video");
        let stream = video.srcObject;

        if (stream === null)
            return;

        // We can't use object literals because that breaks EM_ASM for some reason
        let constraints = {};
        constraints["width"] = $0;
        constraints["height"] = $1;

        stream.getVideoTracks().forEach(track => {
            track.applyConstraints(constraints);
        });

        console.log("[WebCamera] Applied resolution " + $0 + "x" + $1);
    }, size.width, size.height);
    // clang-format on
}
//-----------------------------------------------------------------------------
void WebCamera::close()
{
    // clang-format off
    MAIN_THREAD_EM_ASM({
        let video = document.querySelector("#capture-video");
        let stream = video.srcObject;

        if (stream === null) {
            console.log("[WebCamera] Stream is already closed");
        }

        stream.getVideoTracks().forEach(track => {
            if (track.readyState == "live") {
                track.stop();
                stream.removeTrack(track);
            }
        });

        console.log("[WebCamera] Stream closed");
    });
    // clang-format on
}
//-----------------------------------------------------------------------------
