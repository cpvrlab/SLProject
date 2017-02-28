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
import android.media.Image;
import android.media.ImageReader;
import android.os.IBinder;
import android.support.annotation.NonNull;
import android.util.Log;

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
    public static boolean isTransitioning = false;
    public static boolean isRunning = false;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession captureSession;
    protected ImageReader imageReader;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.i(TAG, "GLES3Camera2Service.onStartCommand flags " + flags + " startId " + startId);

        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            String pickedCamera = getCamera(manager);
            manager.openCamera(pickedCamera, cameraStateCallback, null);
            imageReader = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2);
            imageReader.setOnImageAvailableListener(onImageAvailableListener, null);
            Log.i(TAG, "imageReader created");
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }

        return super.onStartCommand(intent, flags, startId);
    }

    // Return the Camera Id which matches the field videoType
    public String getCamera(CameraManager manager) {
        try {
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
                int cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (cOrientation == videoType)
                    return cameraId;
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
                session.setRepeatingRequest(createCaptureRequest(), null, null);
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
        @Override
        public void onImageAvailable(ImageReader reader) {

            // The opengl renderer runs in its own thread. We have to copy the image in the renderers thread!
            GLES3Lib.view.queueEvent(new Runnable() {
                @Override
                public void run() {
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

                    int Yb = Y.getBuffer().remaining();
                    int Ub = U.getBuffer().remaining();
                    int Vb = V.getBuffer().remaining();

                    /*
                    int yPixstride = Y.getPixelStride();
                    int uPixstride = Y.getPixelStride();
                    int vPixstride = Y.getPixelStride();

                    int yRowstride = Y.getRowStride();
                    int uRowstride = Y.getRowStride();
                    int vRowstride = Y.getRowStride();
                    */

                    byte[] data = new byte[Yb + Ub + Vb];

                    Y.getBuffer().get(data, 0, Yb);
                    U.getBuffer().get(data, Yb, Ub);
                    V.getBuffer().get(data, Yb + Ub, Vb);

                    ///////////////////////////////////////////////////////////////
                    GLES3Lib.copyVideoImage(img.getWidth(), img.getHeight(), data);
                    ///////////////////////////////////////////////////////////////

                    img.close();
                }
            });
        }
    };


    public void actOnReadyCameraDevice() {
        try {
            cameraDevice.createCaptureSession(Arrays.asList(imageReader.getSurface()), sessionStateCallback, null);
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