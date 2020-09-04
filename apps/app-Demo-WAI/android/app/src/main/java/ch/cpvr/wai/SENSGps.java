package ch.cpvr.wai;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.GpsStatus;
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
    SENSGpsStatusListener _statusListener;
    boolean _isRunning = false;

    public void init(Context context) {
        _context = context;
        //_activity = (Activity)context;
        if(_locationManager == null) {
            _locationManager = (LocationManager) _context.getSystemService(Context.LOCATION_SERVICE);
        }
    }

    @SuppressWarnings("ResourceType")
    public void start() {
        Log.i("SENSGps", "start()");

        if(_isRunning)
            return;

        if (_locationManager != null && _locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
            Log.i("SENSGps", "Requesting GPS location updates");

            if (_locationListener == null) {
                _locationListener = new SENSLocationListener();
            }
            if (_statusListener == null) {
                _statusListener = new SENSGpsStatusListener();
            }

            Activity activity = (Activity)_context;
            activity.runOnUiThread(new Runnable() {
                public void run() {
                    if(!_locationManager.addGpsStatusListener(_statusListener)) {
                        Log.i("SENSGps", "addGpsStatusListener failed");
                    }
                    _locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                            1000,
                            0,
                            _locationListener);

                    _isRunning = true;
                }
            });
        } else {
            _isRunning = false;
        }
    }
}

class SENSGpsStatusListener implements GpsStatus.Listener {
    @Override
    public void onGpsStatusChanged(int event) {
        Log.i("SENSGps", "onGpsStatusChanged");
    }
}

class SENSLocationListener implements LocationListener {

    private static final String TAG = "SENSGps";
    protected String latestHdop;
    protected String latestPdop;
    protected String latestVdop;
    protected String geoIdHeight;
    protected String ageOfDgpsData;
    protected String dgpsId;
    protected int satellitesUsedInFix;

    @Override
    public void onLocationChanged(Location loc) {
        Log.i(TAG, "onLocationChanged");
        if (loc != null) {

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