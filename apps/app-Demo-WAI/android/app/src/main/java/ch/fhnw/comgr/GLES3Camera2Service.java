//#############################################################################
//  File:      GLES3Camera2Service.java
//  Author:    Marcus Hudritsch
//  Date:      Spring 2017
//  Purpose:   Android camera2 service implementation
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Please do not change the name space. The SLProject app is identified in the app-store with it.
package ch.fhnw.comgr;

import android.app.Service;
import android.content.Intent;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
//import android.os.Handler;
import android.os.IBinder;
import android.support.annotation.NonNull;
import android.util.Log;
import android.util.Size;
import android.util.SizeF;

import java.util.Arrays;

/**
 * The camera service is started from the activity with cameraStart and
 * stopped with cameraStop. These methods are called from within the views
 * renderer whenever the displayed scene requests a video image.
 * See GLES3View.Renderer.onDrawFrame for the invocation.
 * Camera permission is checked in the activity at startup.
 */
@SuppressWarnings("MissingPermission")
public class GLES3Camera2Service extends Service {
    protected static final String TAG = "SLProject";
    public static int videoType = CameraCharacteristics.LENS_FACING_BACK;
    public static int requestedVideoSizeIndex = -1; // see getRequestedSize
    public static boolean isTransitioning = false;
    public static boolean isRunning = false;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession captureSession;
    protected ImageReader imageReader;

    /*
    // Thread that handles the onImageAvailable (this would be the russian way)
    private Handler GLES_ThreadHandler;
    public GLES3Camera2Service() {GLES_ThreadHandler = GLES3Lib.view.getHandler();}
    */

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i(TAG, "GLES3Camera2Service.onStartCommand flags " + flags + " startId " + startId);

        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            String pickedCamera = getCamera(manager, videoType);
            manager.openCamera(pickedCamera, cameraStateCallback, null);
            Size videoSize = getRequestedSize(manager, videoType, requestedVideoSizeIndex);
            if (videoSize.getWidth() > 0 && videoSize.getHeight() > 0) {
                imageReader = ImageReader.newInstance(videoSize.getWidth(), videoSize.getHeight(), ImageFormat.YUV_420_888, 2);
                imageReader.setOnImageAvailableListener(onImageAvailableListener, null); //GLES_ThreadHandler);
                Log.i(TAG, "imageReader created");
            } else {
                Log.i(TAG, "No imageReader created: videoSize is zero!");
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }

        return super.onStartCommand(intent, flags, startId);
    }

    /**
     * Returns the Camera Id which matches the field lensFacing
     * @param manager The manager got by getSystemService(CAMERA_SERVICE)
     * @param lensFacing LENS_FACING_BACK or LENS_FACING_FRONT
     */
    public String getCamera(CameraManager manager, int lensFacing) {
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                int cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (cOrientation == lensFacing) {

                    float[] lensFocalLengths             = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                    int     lensFocusDistanceCalibration = characteristics.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION);
                    float   lensMinimumFocusDistance     = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
                    float[] lensApertures                = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES);
                    SizeF   sensorPhysicalSize           = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
                    android.graphics.Rect sensorActiveArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);

                    GLES3Lib.view.queueEvent(new Runnable() {
                        @Override
                        public void run() {
                            GLES3Lib.setDeviceParameter("DeviceLensFocalLength",              String.valueOf(lensFocalLengths[0]));
                            GLES3Lib.setDeviceParameter("DeviceLensAperture",                 String.valueOf(lensApertures[0]));
                            GLES3Lib.setDeviceParameter("DeviceLensFocusDistanceCalibration", String.valueOf(lensFocusDistanceCalibration));
                            GLES3Lib.setDeviceParameter("DeviceLensMinimumFocusDistance",     String.valueOf(lensMinimumFocusDistance));
                            GLES3Lib.setDeviceParameter("DeviceSensorPhysicalSizeW",          String.valueOf(sensorPhysicalSize.getWidth()));
                            GLES3Lib.setDeviceParameter("DeviceSensorPhysicalSizeH",          String.valueOf(sensorPhysicalSize.getHeight()));
                            GLES3Lib.setDeviceParameter("DeviceSensorActiveArraySizeW",       String.valueOf(sensorActiveArraySize.width()));
                            GLES3Lib.setDeviceParameter("DeviceSensorActiveArraySizeH",       String.valueOf(sensorActiveArraySize.height()));
                        }
                    });

                    return cameraId;
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        return null;
    }


    /**
     * Returns the requested video size in pixel
     * @param manager The manager got by getSystemService(CAMERA_SERVICE)
     * @param lensFacing LENS_FACING_BACK or LENS_FACING_FRONT
     * @param requestedSizeIndex An index of -1 returns the default size of 640x480
     */
    private Size getRequestedSize(CameraManager manager,
                                  int lensFacing,
                                  int requestedSizeIndex) {

        Size[] sizes = getOutputSizes(manager, lensFacing);

        // On certain old Androids getOutputSizes can return empty arrays
        if (sizes == null || sizes.length == 0)
            return new Size(0,0);

        // set default size index to a size in the middle of the array if 640x480 is not found
        int defaultSizeIndex = sizes.length / 2;

        // get the index of the 640x480 resolution
        for (int i = 0; i < sizes.length; ++i) {
            if (sizes[i].getWidth() == 640 && sizes[i].getHeight() == 480) {
                defaultSizeIndex = i;
                break;
            }
        }

        // pass all resolutions to SLProject
        GLES3Lib.view.queueEvent(new Runnable() {
            @Override
            public void run() {
                for (int i = 0; i < sizes.length; ++i) {
                    GLES3Lib.setCameraSize(i,
                            sizes.length,
                            sizes[i].getWidth(),
                            sizes[i].getHeight());
                }
            }
        });

        if (requestedSizeIndex <= -1)
            return sizes[defaultSizeIndex];
        else if (requestedSizeIndex >= sizes.length)
            return sizes[sizes.length-1];
        else
            return sizes[requestedSizeIndex];
    }

    /**
     * Returns an array of output sizes for the requested camera (front or back)
     * @param manager The manager got by getSystemService(CAMERA_SERVICE)
     * @param lensFacing LENS_FACING_BACK or LENS_FACING_FRONT
     */
    private Size[] getOutputSizes(CameraManager manager, int lensFacing) {
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                int cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (cOrientation == lensFacing) {
                    StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    return streamConfigurationMap.getOutputSizes(ImageFormat.YUV_420_888);
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        return null;
    }

    protected CameraDevice.StateCallback cameraStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            Log.i(TAG, "CameraDevice.StateCallback onOpened");
            cameraDevice = camera;
            actOnReadyCameraDevice();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            Log.w(TAG, "CameraDevice.StateCallback onDisconnected");
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            Log.e(TAG, "CameraDevice.StateCallback onError " + error);
        }
    };

    protected CameraCaptureSession.StateCallback sessionStateCallback = new CameraCaptureSession.StateCallback() {
        @Override
        public void onConfigured(@NonNull CameraCaptureSession session) {
            Log.i(TAG, "CameraCaptureSession.StateCallback onConfigured");
            GLES3Camera2Service.this.captureSession = session;
            try {
                session.setRepeatingRequest(createCaptureRequest(), null, null); //GLES_ThreadHandler);
                isTransitioning = false;
                isRunning = true;
            } catch (CameraAccessException e) {
                Log.e(TAG, e.getMessage());
            }
        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
        }
    };

    protected ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener() {

        // This handler is called in the rendering thread of GLES3View
        @Override
        public void onImageAvailable(ImageReader reader) {

            // The following code withing run() {...} runs in the view thread
            GLES3Lib.view.queueEvent(new Runnable() {
                @Override
                public void run() {

                    // Don't copy the available image if the last wasn't consumed
                    // It can happen that the camera thread pushes to fast to many copy image events
                    // into the view thread so that the view thread only works down the copy image events
                    if (!GLES3Lib.lastVideoImageIsConsumed.get())
                        return;

                    // This avoids the next call into this before the image got displayed
                    GLES3Lib.lastVideoImageIsConsumed.set(false);

                    //Log.i(TAG, "<" + Thread.currentThread().getId());
                    Image img = reader.acquireLatestImage();

                    if (img == null)
                        return;

                    // Check image format
                    int format = reader.getImageFormat();
                    if (format != ImageFormat.YUV_420_888) {
                        throw new IllegalArgumentException("Camera image must have format YUV_420_888.");
                    }

                    Image.Plane[] planes = img.getPlanes();

                    Image.Plane Y = planes[0];
                    Image.Plane U = planes[1];
                    Image.Plane V = planes[2];

                    int ySize = Y.getBuffer().remaining();
                    int uSize = U.getBuffer().remaining();
                    int vSize = V.getBuffer().remaining();

                    byte[] data = new byte[ySize + uSize + vSize];
                    Y.getBuffer().get(data, 0, ySize);
                    U.getBuffer().get(data, ySize, uSize);
                    V.getBuffer().get(data, ySize + uSize, vSize);

                    ///////////////////////////////////////////////////////////////
                    GLES3Lib.copyVideoImage(img.getWidth(), img.getHeight(), data);
                    ///////////////////////////////////////////////////////////////

                    /*
                    This version of the separate copying of the planes is astonishingly not faster!
                    byte[] bufY = new byte[ySize];
                    byte[] bufU = new byte[uSize];
                    byte[] bufV = new byte[vSize];

                    Y.getBuffer().get(bufY, 0, ySize);
                    U.getBuffer().get(bufU, 0, uSize);
                    V.getBuffer().get(bufV, 0, vSize);

                    int yPixStride = Y.getPixelStride();
                    int uPixStride = Y.getPixelStride();
                    int vPixStride = Y.getPixelStride();

                    int yRowStride = Y.getRowStride();
                    int uRowStride = Y.getRowStride();
                    int vRowStride = Y.getRowStride();

                    // For future call of GLES3Lib.copyVideoYUVPlanes
                    GLES3Lib.copyVideoYUVPlanes(img.getWidth(), img.getHeight(),
                                                bufY, ySize, yPixStride, yRowStride,
                                                bufU, uSize, uPixStride, uRowStride,
                                                bufV, vSize, vPixStride, vRowStride);
                    */

                    img.close();

                    // Request a new rendering
                    GLES3Lib.view.requestRender();
                }
            });
        }
    };

    public void actOnReadyCameraDevice() {
        try {
            cameraDevice.createCaptureSession(Arrays.asList(imageReader.getSurface()), sessionStateCallback, null); //GLES_ThreadHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }
    }

    @Override
    public void onDestroy() {
        Log.i(TAG, "GLES3Camera2Service.onDestroy");
        try {
            captureSession.abortCaptures();
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }
        captureSession.close();
        cameraDevice.close();
        imageReader.close();

        isTransitioning = false;
        isRunning = false;
    }

    protected CaptureRequest createCaptureRequest() {
        try {
            CaptureRequest.Builder builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);

            // Turn off auto focus. We want to calibrate with the focus on infinty.
            //builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF);
            //builder.set(CaptureRequest.LENS_FOCUS_DISTANCE,0.0f);

            builder.addTarget(imageReader.getSurface());
            return builder.build();
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
            return null;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}