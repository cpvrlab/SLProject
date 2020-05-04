package ch.cpvr.nativewai;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.NativeActivity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import static android.hardware.camera2.CameraMetadata.LENS_FACING_BACK;

public class WAIActivity extends NativeActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private boolean isCamera2Device() {
        CameraManager camMgr = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        boolean camera2Dev = true;
        try {
            String[] cameraIds = camMgr.getCameraIdList();
            if (cameraIds.length != 0 ) {
                for (String id : cameraIds) {
                    CameraCharacteristics characteristics = camMgr.getCameraCharacteristics(id);
                    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
                    int facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY &&
                            facing == LENS_FACING_BACK) {
                        camera2Dev =  false;
                    }
                }
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
            camera2Dev = false;
        }
        return camera2Dev;
    }

    private static final int PERMISSION_REQUEST_CODE_CAMERA = 1;
    public void RequestCamera() {
        if(!isCamera2Device()) {
            Log.e("WAI", "Found legacy camera Device, this sample needs camera2 device");
            return;
        }
        String[] accessPermissions = new String[] {
                Manifest.permission.CAMERA,
                Manifest.permission.INTERNET
        };
        boolean needRequire  = false;
        for(String access : accessPermissions) {
            int curPermission = ActivityCompat.checkSelfPermission(this, access);
            if(curPermission != PackageManager.PERMISSION_GRANTED) {
                needRequire = true;
                break;
            }
        }
        if (needRequire) {
            ActivityCompat.requestPermissions(
                    this,
                    accessPermissions,
                    PERMISSION_REQUEST_CODE_CAMERA);
            return;
        }
        notifyCameraPermission(true);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        /*
         * if any permission failed, the sample could not play
         */
        if (PERMISSION_REQUEST_CODE_CAMERA != requestCode) {
            super.onRequestPermissionsResult(requestCode,
                    permissions,
                    grantResults);
            return;
        }

        if(grantResults.length == 2) {
            notifyCameraPermission(grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED);
        }
    }

    native static void notifyCameraPermission(boolean granted);

    static {
        System.loadLibrary("native-lib");
    }
}

