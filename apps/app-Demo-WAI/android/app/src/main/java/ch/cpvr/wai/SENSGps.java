package ch.cpvr.wai;

import android.app.Activity;
import android.content.Context;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.location.LocationProvider;
import android.os.Bundle;
import android.util.Log;

//static java gps backend for SENSNdkGps cpp class
public class SENSGps {

    Context _context;
    LocationManager _locationManager;
    SENSLocationListener _locationListener;
    boolean _isRunning = false;

    //set java activity context
    public void init(Context context) {
        _context = context;
    }

    @SuppressWarnings("ResourceType")
    public void start() {
        Log.i("SENSGps", "start()");

        if(_context == null) {
            Log.i("SENSGps", "start: you have to call init first!");
            return;
        }

        if(_isRunning)
            return;

        if(_locationManager == null) {
            _locationManager = (LocationManager) _context.getSystemService(Context.LOCATION_SERVICE);
        }
        if (_locationListener == null) {
            _locationListener = new SENSLocationListener();
        }

        if (_locationManager != null && _locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            Log.i("SENSGps", "Requesting GPS location updates");

            _isRunning = true;

            Activity activity = (Activity)_context;
            activity.runOnUiThread(new Runnable() {
                public void run() {

                    _locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                            1000,
                            0,
                            _locationListener);
                }
            });

        } else {
            _isRunning = false;
        }
    }

    public void stop() {
        if(!_isRunning)
            return;

        if (_locationListener != null) {
            Log.d("SENSGps", "Removing locationManager updates");

            Activity activity = (Activity)_context;
            activity.runOnUiThread(new Runnable() {
                public void run() {
                    _locationManager.removeUpdates(_locationListener);
                }
            });
        }
        _isRunning = false;
    }

    native static void onLocationLLA(double latitudeDEG, double longitudeDEG, double altitudeM, float accuracyM);
}

class SENSLocationListener implements LocationListener {

    private static final String TAG = "SENSGps";

    @Override
    public void onLocationChanged(Location loc) {
        Log.i(TAG, "onLocationChanged");
        if (loc != null) {

            SENSGps.onLocationLLA(loc.getLatitude(),
                    loc.getLongitude(),
                    loc.getAltitude(),
                    loc.getAccuracy());
        }
    }

    @Override
    public void onProviderDisabled(String provider) {
        Log.i(TAG, "onProviderDisabled");
    }

    @Override
    public void onProviderEnabled(String provider) {
        Log.i(TAG, "onProviderEnabled");
    }

    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {
        if (status == LocationProvider.OUT_OF_SERVICE) {
            Log.i(TAG, provider + " is out of service");
        }

        if (status == LocationProvider.AVAILABLE) {
            Log.i(TAG, provider + " is available");
        }

        if (status == LocationProvider.TEMPORARILY_UNAVAILABLE) {
            Log.i(TAG, "onStatusChanged:" + provider + " is temporarily unavailable");
        }
    }
}