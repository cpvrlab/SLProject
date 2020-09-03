package ch.cpvr.wai;

import android.content.Context;
import android.location.LocationManager;
import android.util.Log;
import androidx.annotation.Keep;

//static java gps backend for SENSNdkGps cpp class
public class SENSGps {

    static private Context _context;
    static private LocationManager _locationManager;

    public static void setContext(Context context) {
        _context = context;
        _locationManager = (LocationManager)_context.getSystemService(Context.LOCATION_SERVICE);
    }

    @Keep
    public static void start() {
        Log.i("SENSGps", "start()");
        //_locationManager = (LocationManager)_context.getSystemService(Context.LOCATION_SERVICE);
        boolean provided = _locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
        if(provided)
            Log.i("SENSGps", "provided");
    }


}