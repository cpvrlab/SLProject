package ch.bfh.ar;


import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.util.Log;
import android.util.Size;

import ch.bfh.ar.camera.PreviewCallback;

@SuppressWarnings("MissingPermission")
public class CameraService extends Service {
    private static final String TAG = "SLProject:Cam";

    private PreviewCallback previewCallback;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        try {
            CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            String cameraId = cameraManager.getCameraIdList()[0];
            CameraCharacteristics cc = cameraManager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap streamConfigs = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] outputSizes = streamConfigs.getOutputSizes(ImageFormat.RAW_SENSOR);

            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    try {
                        ImageAvailableListener imageAvailableListener = new ImageAvailableListener();
                        previewCallback = new PreviewCallback(outputSizes, imageAvailableListener, camera);
                        previewCallback.startPreviewSession();
                    } catch (CameraAccessException e) {
                        Log.e(TAG, "Cannot access to cam");
                    }
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                    previewCallback.cancelActiveCaptureSession();
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                }
            }, null);


        } catch (CameraAccessException e) {
            Log.e(TAG, "Cannot setup camera");
        }

        return super.onStartCommand(intent, flags, startId);
    }


    public class ImageAvailableListener implements ImageReader.OnImageAvailableListener {

        @Override
        public void onImageAvailable(ImageReader reader) {
            Log.i(TAG, "Image available");
            Image image = reader.acquireLatestImage();

            // TODO: Receive image in form of cv::Mat

            //GLES3Lib.passImageMat();
            image.close();
        }
    }

    @Override
    public void onDestroy() {
        previewCallback.cancelActiveCaptureSession();
        super.onDestroy();
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
