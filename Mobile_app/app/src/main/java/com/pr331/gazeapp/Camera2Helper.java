package com.pr331.gazeapp;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;

import com.google.mediapipe.components.CameraHelper;

import java.io.File;
import java.util.Arrays;

//import javax.annotation.Nullable;

public class Camera2Helper extends CameraHelper {
    public static final String TAG = "Camera2Helper";
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;

    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private ImageReader imageReader;
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private SurfaceTexture outputSurface;
    private Size frameSize;
    private int frameRotation;
    private CameraHelper.CameraFacing cameraFacing;
    private Context context;
    private Size imageSize;
    private Size focalLengthPixels;

    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    @Nullable private CameraCharacteristics usingCameraCharacteristics = null;

    public Camera2Helper(Context context) {
        this.context = context;
    }

    public Camera2Helper(Context context, SurfaceTexture surfaceTexture, int rotation) {
        this.context = context;
        this.outputSurface = surfaceTexture;
        this.frameRotation = (ORIENTATIONS.get(rotation) + 270) % 360;
    }

    @Override
    public void startCamera(Activity context, CameraFacing cameraFacing, @Nullable SurfaceTexture surfaceTexture) {
        this.cameraFacing = cameraFacing;
        closeCamera();
        startBackgroundThread();
        openCamera();

    }

    @Override
    public Size computeDisplaySizeFromViewSize(Size viewSize) {
        if (viewSize == null || frameSize == null) {
            // Wait for all inputs before setting display size.
            Log.d(TAG, "viewSize or frameSize is null.");
            return null;
        }

        // Valid rotation values are 0, 90, 180 and 270.
        // Frames are rotated relative to the device's "natural" landscape orientation. When in portrait
        // mode, valid rotation values are 90 or 270, and the width/height should be swapped to
        // calculate aspect ratio.
        Log.v(TAG, "CheckFrame frameRatio is: " + frameRotation);
        Log.v(TAG, "Check frameSize is: " + frameSize);
        float frameAspectRatio =    // 1.777777778
                frameRotation == 0 || frameRotation == 180
                        ? frameSize.getHeight() / (float) frameSize.getWidth()
                        : frameSize.getWidth() / (float) frameSize.getHeight();

        float viewAspectRatio = viewSize.getWidth() / (float) viewSize.getHeight(); //1.649484536

        // Match shortest sides together.
        int scaledWidth;
        int scaledHeight;
//        scaledWidth = viewSize.getWidth();
//        scaledHeight = viewSize.getHeight();
        if (frameAspectRatio < viewAspectRatio) {
            scaledWidth = viewSize.getWidth();
            scaledHeight = Math.round(viewSize.getWidth() / frameAspectRatio);
        } else {
            scaledHeight = viewSize.getHeight();
            scaledWidth = Math.round(viewSize.getHeight() * frameAspectRatio);
        }

        return new Size(scaledWidth, scaledHeight);
    }

    @Override
    public boolean isCameraRotated() {
        return frameRotation % 180 == 0;
    }

    final CameraCaptureSession.CaptureCallback captureCallbackListener = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
            super.onCaptureCompleted(session, request, result);
            Toast.makeText(context, "Saved:" + file, Toast.LENGTH_SHORT).show();
            createCameraPreview();
        }
    };

    private void closeCamera() {
        try {
            stopBackgroundThread();
            Log.d(TAG, "Closing camera ");
            if (null != cameraDevice) {
                cameraDevice.close();
                cameraDevice = null;
            }
            if (null != imageReader) {
                imageReader.close();
                imageReader = null;
            }
        } catch (Exception e) {
            Log.d(TAG, e.toString());
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraId = null;
            for (String cameraIdLoop : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraIdLoop);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null) {
                    if ((facing == CameraCharacteristics.LENS_FACING_BACK && cameraFacing == CameraFacing.BACK) || (facing == CameraCharacteristics.LENS_FACING_FRONT && cameraFacing == CameraFacing.FRONT)) {
                        cameraId = cameraIdLoop;
                        Log.e(TAG, "Open camera " + facing + " " + cameraId);

                        StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                        assert map != null;
                        Size tmp_imageDimension = map.getOutputSizes(SurfaceTexture.class)[1]; //[0];
//                        Log.v(TAG, "CheckFrame sensorOrientation: " + frameRotation);
//                        if (frameRotation % 180 == 90) {
//                            imageDimension = new Size(tmp_imageDimension.getHeight(), tmp_imageDimension.getWidth());
//                            Log.v(TAG, "CheckFrame swap Dimenssion: " + imageDimension.getWidth() + " " + imageDimension.getHeight());
//                        }
//                        else    {
//                            imageDimension = tmp_imageDimension;
//                            Log.v(TAG, "CheckFrame keep Dimenssion: " + imageDimension.getWidth() + " " + imageDimension.getHeight());
//                        }

                        imageDimension = tmp_imageDimension; //new Size(tmp_imageDimension.getWidth()/1, tmp_imageDimension.getHeight() / 1);
                        imageSize = imageDimension;

                        Log.v(TAG, "Image size " + imageSize);

                        // Add permission for camera and let user grant the permission
                        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                            ActivityCompat.requestPermissions((Activity) context, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                            Log.d(TAG, "Permission issue");
                            return;
                        }
                        Log.v(TAG, "CheckFocal: imageDimension " + imageDimension.getWidth() + " " + imageDimension.getHeight() + " frameRotation " + frameRotation);
                        usingCameraCharacteristics = characteristics;

                        if (frameRotation % 180 == 90) {
                            focalLengthPixels = calculateFocalLengthInPixels(imageDimension.getWidth(), imageDimension.getHeight());
                        }
                        else    {
                            focalLengthPixels = calculateFocalLengthInPixels(imageDimension.getHeight(), imageDimension.getWidth());
                        }

                        float[] intrinsic = new float[5];
                        intrinsic = usingCameraCharacteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
                        Log.d(TAG, "Intrinsic parameters: " + Arrays.toString(intrinsic));

                        Log.d(TAG, "Opening camera from manager " + cameraId);
                        manager.openCamera(cameraId, stateCallback, null);

                    }
                }
            }

            //fetchFrame();
            /*if(previewDisplayView!=null){
                previewDisplayView.setVisibility(View.VISIBLE);
            }*/
           /* if(outputTextureView!=null){
                outputTextureView.setVisibility(View.VISIBLE);
                initRecorder();
            }*/
            Log.d(TAG, "Camera debug start 113");

        } catch (CameraAccessException e) {
            e.printStackTrace();
            Log.d(TAG, e.toString());
        }

    }

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            //This is called when the camera is open
            Log.e(TAG, "onOpened");
            cameraDevice = camera;
            createCameraPreview();
            if (onCameraStartedListener != null) {
                onCameraStartedListener.onCameraStarted(outputSurface);

            }

        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            try {
                Log.d(TAG, " Error on CameraDevice ");
                cameraDevice.close();
                cameraDevice = null;
            } catch (Exception e) {
                Log.d(TAG, "ERROR: " + e.toString() + " ER " + error);
                e.printStackTrace();
            }
        }
    };

    private File file;

    //private CameraCaptureSession cameraCaptureSession;
    // @RequiresApi(api = Build.VERSION_CODES.O)
    protected void createCameraPreview() {
        try {
            Log.d(TAG, "Creating camera preview");
            outputSurface = (outputSurface == null) ? new CustomSurfaceTexture(0) : outputSurface;

            SurfaceTexture texture = outputSurface;//textureView.getSurfaceTexture();
//            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
//            texture.setDefaultBufferSize(imageDimension.getHeight(), imageDimension.getWidth());
            frameSize = imageDimension;
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            //captureRequestBuilder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_ON);
            captureRequestBuilder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON);
//            captureRequestBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(frameRotation));
            captureRequestBuilder.addTarget(surface);
            cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    //The camera is already closed
                    if (null == cameraDevice) {
                        return;
                    }
                    // When the session is ready, we start displaying the preview.
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(context, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
            Log.d(TAG, e.toString());
        }
    }

    private int getOrientation(int rotation) {
        // Sensor orientation is 90 for most devices, or 270 for some devices (eg. Nexus 5X)
        // We have to take that into account and rotate JPEG properly.
        // For devices with orientation of 90, we simply return our mapping from ORIENTATIONS.
        // For devices with orientation of 270, we need to rotate the JPEG 180 degrees.
        int mSensorOrientation = 0;//usingCameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        return (rotation + mSensorOrientation + 270) % 360;
    }

    protected void updatePreview() {
        if (null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
//        captureRequestBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(frameRotation));
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public Size getFocalLengthPixels() {
        return focalLengthPixels;
    }

    private Size calculateFocalLengthInPixels(float frameSizeWidth, float frameSizeHeight)   {
        // Computes the focal length of the camera in pixels based on lens and sensor properties
        assert usingCameraCharacteristics != null;
        float sensorWidthMm = usingCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE).getWidth();
        float sensorHeightMm = usingCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE).getHeight();

        float focalLengthPixelsHeight = frameSizeHeight * usingCameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)[0] / sensorHeightMm;
        float focalLengthPixelsWidth = frameSizeWidth * usingCameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)[0] / sensorWidthMm;
        return new Size(Math.round(focalLengthPixelsWidth), Math.round(focalLengthPixelsHeight));
    }

    public CameraCharacteristics getUsingCameraCharacteristics()    {
        return usingCameraCharacteristics;
    }

    public Size getImageSize()  {
        return imageSize;
    }
}
