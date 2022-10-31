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
void WebCamera::open()
{
    // clang-format off
    MAIN_THREAD_EM_ASM({
        console.log("[WebCamera] Requesting stream...");

        // We can't use object literals because that breaks EM_ASM for some reason
        let videoConstraints = {};
        videoConstraints["width"] = 1280;
        videoConstraints["height"] = 720;
        let constraints = { "video": videoConstraints };

        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                console.log("[WebCamera] Stream acquired");

                let video = document.querySelector("#capture-video");
                video.srcObject = stream;
            })
            .catch(error => {
                console.log("[WebCamera] Failed to acquire stream");
                console.log(error);
            });
    });
    // clang-format on

    _isOpened = true;
}
//-----------------------------------------------------------------------------
CVMat WebCamera::read()
{
    if (!_isImageAllocated)
    {
        int32_t width;
        int32_t height;

        // clang-format off
        MAIN_THREAD_EM_ASM({
            let video = document.querySelector("#capture-video");
            let width = video.videoWidth;
            let height = video.videoHeight;

            if (width != 0 && height != 0)
                console.log("[WebCamera] Resolution: " + width + "x" + height);

            setValue($0, video.videoWidth, "i32");
            setValue($1, video.videoHeight, "i32");
        }, &width, &height);
        // clang-format on

        // If the width or the height is zero, the video is not ready
        if (width == 0 || height == 0)
            return CVMat(0, 0, CV_8UC4);

        // Allocate the image data
        uchar* data       = new uchar[4 * (int64_t)width * (int64_t)height];
        _image            = CVMat(height, width, CV_8UC4, data);
        _isImageAllocated = true;
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

    return _image.clone();
}
//-----------------------------------------------------------------------------
void WebCamera::close()
{
    // clang-format off
    MAIN_THREAD_EM_ASM({
        let video = document.querySelector("#capture-video");
        let stream = video.srcObject;

        if (stream !== null) {
            stream.getVideoTracks().forEach(track => {
                if (track.readyState == "live") {
                    track.stop();
                    stream.removeTrack(track);
                }
            });
        }

        console.log("[WebCamera] Tracks closed");
    });
    // clang-format on

    delete[] _image.data;
    _isImageAllocated = false;
}
//-----------------------------------------------------------------------------
