package com.pr331.gazeapp

import android.annotation.SuppressLint
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.view.*
import android.view.View.*
import android.widget.RelativeLayout
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.components.CameraHelper.CameraFacing
import com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.components.PermissionHelper
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.AndroidPacketGetter
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.pr331.gazeapp.ml.IrisLandmark
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*


class MainActivity : AppCompatActivity() {
    private val BINARY_GRAPH_NAME = "iris_tracking_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "output_video"
    // Use for iris tracking
    private val FOCAL_LENGTH_STREAM_NAME = "focal_length_pixel"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "face_landmarks_with_iris"
    private val OUTPUT_LEFT_IRIS_DEPTH_STREAM_NAME = "left_iris_depth_mm"
    private val OUTPUT_RIGHT_IRIS_DEPTH_STREAM_NAME = "right_iris_depth_mm"
    private val NUM_BUFFERS = 2

    private var haveAddedSidePackets = false

    private var leftIrisDepth = 0.0f
    private var rightIrisDepth = 0.0f
    private var landmarks: NormalizedLandmarkList? = null
    private val rgbMap = mutableMapOf<String, Bitmap>()
    private val leftDepth = mutableMapOf<String, Float>()
    private val rightDepth = mutableMapOf<String, Float>()

    private val CAMERA_FACING = CameraFacing.FRONT

    //INITIALIZE GLOBAL CALIBRATION VARIABLES
    private var isCalibrated = false
    private var calibrationStarted = false
    private var calibrationStopped = false

    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private val FLIP_FRAMES_VERTICALLY = true

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private var previewFrameTexture: CustomSurfaceTexture? = null

    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private lateinit var previewDisplayView: SurfaceView

    private lateinit var currentGazeDraw: TextView
    private lateinit var touchPositionDraw: TextView
    private lateinit var touchPositionTextbox: TextView
    private lateinit var gazePositionTextbox: TextView
    private lateinit var currenGazePositionTextbox: TextView
    private lateinit var previewGaze: RelativeLayout

    private lateinit var calibrationDraw: TextView
    private lateinit var irisInputFeature: TensorBuffer
    private lateinit var irisModel: IrisLandmark
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    // GET THE OUTPUT FROM MEDIAPOIPE USING THE PROCESSOR
    // SENDS CAMERAS FRAMES PREPARED BY THE {CONVERTER} TO THE MEDIAPIPE GRAPH AND RUNS THE GRAPH,
    // PREPARES THE OUTPUT AND THEN UPDATES THE {PREVIEWDISPLAYVIEW} TO DISPLAY THE OUTPUT
    private var processor: FrameProcessor? = null
    private val scaleInput = 0.25

    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private var converter: ExternalTextureConverter? = null
    // Creates and manages an {@link EGLContext} -> used with ExternalTextureConverter
    private var eglManager: EglManager? = null

    // Handles camera access via the {@link CameraX} Jetpack support library.
    private var cameraHelper: Camera2Helper? = null
    private var textureName = 65

    private lateinit var focalLength: Size
    private lateinit var displaySize: Size
    private var matrixPixels2World = Matrix()
    //    private var saved_Image = 0
    private var latestKey: String = "-1"

    private var displayDpi: Double = 0.0
    private var displayWidth: Int = 0
    private var displayHeight: Int = 0

    private var xCoeffList: MutableList<Double> = mutableListOf()
    private var yCoeffList: MutableList<Double> = mutableListOf()

    private var xCurrentIris = 0.0
    private var yCurrentIris = 0.0

    private var rightEyeWidth = 0.0
    private var rightEyeHeight = 0.0

    private var xRefList: MutableList<Double> = mutableListOf()
    private var yRefList: MutableList<Double> = mutableListOf()

    //INITIALIZE THE X,Y OF THE GAZE AT THE CENTER OF THE IMAGE (CURRENTBITMAP SIZE) : ( 1920 X 1080)
    private var xCurrentGaze: Double = 960.0
    private var yCurrentGaze: Double = 540.0

    // DEVICE HORIZONTAL FLIP
//    private var xCurrentGaze: Double = 540.0
//    private var yCurrentGaze: Double = 960.0

    private var flag = false

    private var xReferencePoint = 0.0

    //Iris_landmark output
    private var iris: FloatArray = floatArrayOf()
    private var     eyeContour: FloatArray = floatArrayOf()
    private val gazeSmoothing = GazeSmoothing(12);


    @SuppressLint("SimpleDateFormat")
    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        supportActionBar?.hide()

        previewDisplayView = SurfaceView(this)
        setupPreviewDisplayView()

        touchPositionDraw = findViewById(R.id.touch_draw)
        touchPositionTextbox = findViewById(R.id.touch_pos_textbox)
        gazePositionTextbox = findViewById(R.id.gaze_position_textbox)
        currenGazePositionTextbox = findViewById(R.id.current_gaze_pos_textbox)
        currentGazeDraw = findViewById(R.id.current_gaze_draw)
        previewGaze = findViewById(R.id.preview_draw_gaze)
        calibrationDraw = findViewById(R.id.calibration_draw)

        irisModel = IrisLandmark.newInstance(this)
        irisInputFeature =TensorBuffer.createFixedSize(intArrayOf(1, 64, 64, 3), DataType.FLOAT32)

        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this)

        eglManager = EglManager(null)
        // INITIALIZED IN HERE BUT GETS THE ACTUAL INPUT FROM THE {CONVERTER} ONRESUME[1]
        processor= FrameProcessor(
            this,
            eglManager!!.getNativeContext(),
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )
        processor!!.videoSurfaceOutput.setFlipY(FLIP_FRAMES_VERTICALLY)
        processor!!.addPacketCallback(OUTPUT_LANDMARKS_STREAM_NAME) { packet ->
            val landmarksRaw = PacketGetter.getProtoBytes(packet)
            // Iris index (10 points): 468 to 478, (center is 468 and 473)
            landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw)
            if (landmarks == null) {
                return@addPacketCallback
            }
            // ? (WITHOUT SCALE INPUT) CurrentBitmap: WIDTH: 480, HEIGHT: 270
            val currentBitmap = rgbMap[packet.timestamp.toString()]
            Log.i("check", "current bitmap width:${currentBitmap?.width}")
            if (currentBitmap != null) {
                // THE WIDTH AND HEIGHT OF THE IMAGE, displayWidth: 1920, displayHeight, 1080
                displayWidth = (currentBitmap.width / scaleInput).toInt()
                displayHeight = (currentBitmap.height / scaleInput).toInt()
                // ? THE RECT IS APPLIED ON THE currentBitmap: WIDTH:480, HEIGHT: 270
                val leftRect = ImageUtil().getRect(
                    landmarks!!,
//                    listOf(23, 27, 130, 133),
                    listOf(71, 9, 116, 195),
                    currentBitmap.width,
                    currentBitmap.height
                )

//                val rightRect = ImageUtil().getRect(
//                    landmarks!!,
//                    listOf(9, 195, 345, 298),
//                    currentBitmap.width,
//                    currentBitmap.height
//                )
                // THIS IS A SCOPED FUNCTION -> EXECUTING A BLOCK OF CODE ON AN OBJECT
                // (IN THIS CASE A BITMAP) AND RETURN A LAMBDA RESULT)
                val leftEyeBitmap = rgbMap[packet.timestamp.toString()]?.let {
                    ImageUtil().cropResizePad(
                        it,
                        leftRect,
                        64,
                        64,
                        false
                    )
                }
//                val rightEyeBitmap = rgbMap[packet.timestamp.toString()]?.let {
//                    ImageUtil().cropResizePad(
//                        it,
//                        rightRect,
//                        64,
//                        64,
//                        false
//                    )
//                }
                if(leftEyeBitmap!=null) {
//                    val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
//                    ImageUtil().saveImage(leftEyeBitmap, "$timestamp.jpeg")
                    val outputVector = predictIrisLandmark(leftEyeBitmap)

                    eyeContour = outputVector.component1()
//                    val x1 = eyeContour[24]
//                    val x2 = eyeContour[75]
//                    val y3 = eyeContour[64]
//                    val y4 = eyeContour[82]
                    iris = outputVector.component2()
                }

                val xIrisCenter = iris[0]
                val yIrisCenter = iris[1]
                if (calibrationStarted && !calibrationStopped){
                    xReferencePoint = (((eyeContour[24] + 2) - (eyeContour[75] - 2))/2 + (eyeContour[75] - 2)).toDouble()
                    yReferencePoint = (((eyeContour[64] + 2) - (eyeContour[82] - 2))/2 + (eyeContour[82] - 2)).toDouble()

                    val xCoeffs = xIrisCenter - xReferencePoint
                    val yCoeffs = yIrisCenter - yReferencePoint
                    xCoeffList.add(xCoeffs)
                    yCoeffList.add(yCoeffs)
                }
                // NEED TO RUN THIS BLOCK OF CODE ONLY ONCE, JUST TO GET THE AVERAGE
                if (flag) {
                    xCurrentIris = xCoeffList.average() + xReferencePoint
                    yCurrentIris = yCoeffList.average() + yReferencePoint
                }
                if (calibrationStopped){
                    flag = false
                    val xDiff = xIrisCenter - xCurrentIris
                    val yDiff = yIrisCenter - yCurrentIris

                    xCurrentIris += xDiff
                    yCurrentIris += yDiff
                    Log.i (TAG_MAIN, "leftIrisDepth$leftIrisDepth")
                    val xScaleGaze = listOf(380, 420, 450, 500)
                    val yScaleGaze = listOf(380, 420, 450, 500)
                    if (leftIrisDepth in 100.0..350.0) {
                        xCurrentGaze += xDiff * xScaleGaze[0]
                        yCurrentGaze += yDiff * yScaleGaze[0]
                    }
                    else if( leftIrisDepth in 350.0..450.0){
                        xCurrentGaze += xDiff * xScaleGaze[1]
                        yCurrentGaze += yDiff * yScaleGaze[1]
                    }
                    else if (leftIrisDepth in 450.0..550.0){
                        xCurrentGaze += xDiff * xScaleGaze[2]
                        yCurrentGaze += yDiff * yScaleGaze[2]
                    }
                    else {
                        xCurrentGaze += xDiff * xScaleGaze[3]
                        yCurrentGaze += yDiff * yScaleGaze[3]
                    }
//                    val currentGazeCoord = doubleArrayOf(xCurrentGaze, yCurrentGaze)
                    val smoothedGaze = gazeSmoothing.makeGazeSmooth(listOf(listOf(xCurrentGaze,yCurrentGaze), listOf(xCurrentGaze,yCurrentGaze)))
                    drawGazeLocation(smoothedGaze)
                }
            }
        }
        //trailing lambda
        processor!!.addPacketCallback(
            OUTPUT_LEFT_IRIS_DEPTH_STREAM_NAME
        ) { packet: Packet? ->
            leftIrisDepth = PacketGetter.getFloat32(packet)
            if (packet != null) {
                leftDepth[packet.timestamp.toString()] = leftIrisDepth
            }
        }
        processor!!.addPacketCallback(
            OUTPUT_RIGHT_IRIS_DEPTH_STREAM_NAME
        ) { packet ->
            rightIrisDepth = PacketGetter.getFloat32(packet)
            if (packet != null) {
                rightDepth[packet.timestamp.toString()] = rightIrisDepth
            }
        }
        processor!!.addPacketCallback("input_video_cpu") { packet ->
            //INPUT_CPU_IMAGE SIZE IN HERE IS 2069X1164
            var input_cpu_image = AndroidPacketGetter.getBitmapFromRgba(packet)
            input_cpu_image = Bitmap.createScaledBitmap(input_cpu_image, 1920, 1080, false)

            if (packet != null) {
                val rgbMap_put = Bitmap.createScaledBitmap(
                    input_cpu_image,
                    (input_cpu_image.width ),
                    (input_cpu_image.height),
                    true
                )
                rgbMap[packet.timestamp.toString()] = rgbMap_put

                if (latestKey == "-1") {
                    latestKey = packet.timestamp.toString()
                } else {
                    latestKey = maxOf(latestKey, packet.timestamp.toString())
                }

                input_cpu_image.recycle()
            }
            if (latestKey != "-1" && rgbMap.isNotEmpty()) {
                val listKeys = rgbMap.keys.toList()
                if (listKeys.minOrNull()!! <= latestKey) {

                    listKeys.forEach { k ->
                        if (k < latestKey) {
                            rgbMap.remove(k)
                        }
                        else{
                            return@forEach
                        }
                    }
                }
            }
        }
        PermissionHelper.checkAndRequestCameraPermissions(this)
    }
    // KEEP THIS FUNCTION CLOSED
    override fun onResume() {
        super.onResume()

        calibrationDraw.visibility = VISIBLE
        calibrationDraw.x = (1920 / 2).toFloat()
        calibrationDraw.y = (1080 / 2).toFloat()

        converter = ExternalTextureConverter(eglManager?.context, NUM_BUFFERS)

        if (PermissionHelper.cameraPermissionsGranted(this)) {
            var rotation: Int = 0
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                rotation = this.display!!.rotation
            } else {
                rotation = this.windowManager.defaultDisplay.rotation
            }
            converter!!.setRotation(rotation)
            converter!!.setFlipY(FLIP_FRAMES_VERTICALLY)
            startCamera(rotation)

            if (!haveAddedSidePackets) {
                val packetCreator = processor!!.getPacketCreator();
                val inputSidePackets = mutableMapOf<String, Packet>()

                focalLength = cameraHelper?.focalLengthPixels!!

                inputSidePackets[FOCAL_LENGTH_STREAM_NAME] = packetCreator.createFloat32(focalLength.width.toFloat())
                processor!!.setInputSidePackets(inputSidePackets)
                haveAddedSidePackets = true

                val imageSize = cameraHelper!!.imageSize
                val calibrateMatrix = Matrix()
                calibrateMatrix.setValues(
                    floatArrayOf(
                        focalLength.width * 1.0f,
                        0.0f,
                        imageSize.width / 2.0f,
                        0.0f,
                        focalLength.height * 1.0f,
                        imageSize.height / 2.0f,
                        0.0f,
                        0.0f,
                        1.0f
                    )
                )
                val isInvert = calibrateMatrix.invert(matrixPixels2World)
                if (!isInvert) {
                    matrixPixels2World = Matrix()
                }
            }
            val dm = DisplayMetrics()
            windowManager.defaultDisplay.getMetrics(dm)
//            displayPixel_x = dm.widthPixels
//            displayPixel_y = dm.heightPixels
            displayDpi = dm.xdpi.toDouble()


            // THE PROCESSOR NEEDS TO CONSUME THE CONVERTED FRAMES FROM THE {CONVERTER} FOR PROCESSING
            // PROCESSOR GETS THE INPUT IN HERE [1]
            converter!!.setConsumer(processor)
        }
    }
    // KEEP THIS FUNCTION CLOSED
    override fun onPause() {
        super.onPause()
        converter!!.close()

        // Hide preview display until we re-open the camera again.
        previewDisplayView.setVisibility(GONE);

        rgbMap.clear()
        leftDepth.clear()
        rightDepth.clear()
        latestKey = "-1"

        xCoeffList.clear()
        yCoeffList.clear()
        haveAddedSidePackets = false
        calibrationDraw.visibility = INVISIBLE
        currentGazeDraw.visibility = INVISIBLE
        isCalibrated = false
        calibrationStarted = false
        calibrationStopped = false
    }
    // Return false just neutralize the statement
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!PermissionHelper.cameraPermissionsGranted(this)) {
            PermissionHelper.checkAndRequestCameraPermissions(this)
            return true
        }
        // STOP THE CALIBRATION
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (isCalibrated) {
                calibrationDraw.visibility = GONE

                touchPositionDraw.visibility = VISIBLE
                touchPositionDraw.x = event.rawX + previewGaze.x
                touchPositionDraw.y = event.rawY + previewGaze.y

                touchPositionTextbox.text = "Touch position: ${event.rawX}, ${event.rawY}"

                calibrationStopped = true
                flag = true

                return true
            }
        }

        // START THE CALIBRATION
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (!isCalibrated) {

                isCalibrated = true
                calibrationStarted = true
            }
        }
        return true
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
    private fun setupPreviewDisplayView() {
        // Draw image preview
        previewDisplayView.visibility = GONE
        val container = findViewById<ViewGroup>(R.id.preview_display_layout)
        container.addView(previewDisplayView)
        previewDisplayView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) {
                val viewSize = Size(width, height)
                Log.d(TAG_MAIN, "Viewsize $viewSize")

                //THIS DISPLAY SIZE IS SEND TO THE CONVERTER?? PERHAPS SENDING THE WIDTH AND HEIGHT
                // FROM THE VIEW SIZE?
                displaySize = cameraHelper?.computeDisplaySizeFromViewSize(viewSize)
                    ?: Size(1280, 720)
                Log.d(TAG_MAIN, "displaySize: $displaySize")

                //HERE FRAMES ARE PASSED TO THE MEDIAPIPE USING THE MEDIAPIPE CONVERTER
                // connect the converter to the camera-preview frames as its input (via
                //previewFrameTexture), and configure the output width and height as the computed
                // display size.
                converter!!.setSurfaceTextureAndAttachToGLContext(
                    previewFrameTexture, displaySize.width/4,
                    displaySize.height/4
                )
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.d(TAG_MAIN, "Destroyed surface")
                processor?.videoSurfaceOutput?.setSurface(null)
            }
            //OUT THE GRAPH OUTPUT IN HERE (ITS INSIDE THE SURFACEHOLDER.CALLBACK)
            override fun surfaceCreated(holder: SurfaceHolder) {
                processor?.videoSurfaceOutput?.setSurface(null)
            }
        })
    }
    private fun startCamera(rotation: Int) {
        cameraHelper = Camera2Helper(this, CustomSurfaceTexture(textureName), rotation)
//        cameraHelper!!.setFrameRotation(rotation)

        cameraHelper!!.setOnCameraStartedListener(
            OnCameraStartedListener {
                //it IS THE SURFACETEXTURE PASSED AS A PARAMETER
                //THIS IS SURFACE TEXTURE -> HOLD CAMERA FRAMES
                //SURFACE TEXTURE: captures image frames from a stream as an OpenGL ES texture
                //TO use a mediapipe graph, frames captured from the camera should be stored in a
                //regular OpenGL texture object. We use mediapipe's ExternalTextureConverter to convert
                // the image stored in the SurfaceTexture object to a regular OpenGL texture object
                previewFrameTexture = it as CustomSurfaceTexture
                // Make the display view visible to start showing the preview. This triggers the
                // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
                //THIS IS SURFACE VIEW -> DISPLAY CAMERA FRAMES
                previewDisplayView.setVisibility(VISIBLE)
            }
        )
        //NORMALLY IF WE PASS THE previewFrameTexture IT WILL SHOW THE CAMERA FRAMES, BUT IN HERE,
        //THE PREVIEWFRAME IS PASS BY THE MEDIAPIPE USING THE SURFACE HOLDER CALLBACK
        cameraHelper!!.startCamera(this, CAMERA_FACING,  /*surfaceTexture=*/null)

    }
    private fun drawGazeLocation(irisCenterCoordinate: List<Double>) {
        val scr_xleft: Double = irisCenterCoordinate[0]
        val scr_yleft: Double = irisCenterCoordinate[1]
        runOnUiThread {

            currentGazeDraw.x = scr_xleft.toFloat()
            currentGazeDraw.y = scr_yleft.toFloat()
            currentGazeDraw.visibility = VISIBLE
            currentGazeDraw.bringToFront()

            currenGazePositionTextbox.text = "Current Gaze: ${String.format("%.2f", scr_xleft)}, " +
                    "${String.format("%.2f", scr_yleft)}  "
        }
    }
    /**
     * Scale the image to a byteBuffer of [-1,1] values.
     */
    private fun initInputArray(bitmap: Bitmap, inputChannelsImage: Int = 1): ByteBuffer {
        val bytesPerChannel = 4
        val inputChannels = inputChannelsImage
        val batchSize = 1
        val inputBuffer = ByteBuffer.allocateDirect(
                batchSize * bytesPerChannel * bitmap.height * bitmap.width * inputChannels
        )
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val mean = 0.0f
        val std = 255.0f
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixelValue in intValues) {
            if (inputChannels == 1) {
                inputBuffer.putFloat(Color.red(pixelValue).toFloat())
                // inputBuffer.putFloat(pixelValue and 0xFF)   ?is it correct, may be correct :v
            } else {
                inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) - mean) / std)
                inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) - mean) / std)
                inputBuffer.putFloat(((pixelValue and 0xFF) - mean) / std)
            }
        }

        return inputBuffer
    }

    private fun predictIrisLandmark(bitmap: Bitmap): List<FloatArray> {
        val byteBuffer = initInputArray(bitmap, 3)
        irisInputFeature.loadBuffer(byteBuffer)
        // Runs model inference and gets result.
        val outputs = irisModel.process(irisInputFeature)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val outputFeature1 = outputs.outputFeature1AsTensorBuffer
        val outLandmark0 = outputFeature0.floatArray
        val outLandmark1 = outputFeature1.floatArray
        return listOf(outLandmark0, outLandmark1)
    }

    override fun onBackPressed() {
        super.onBackPressed()
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
            System.loadLibrary("mediapipe_jni")
            try {
                System.loadLibrary("opencv_java3")
            } catch (e: UnsatisfiedLinkError) {
                // Some example apps (e.g. template matching) require OpenCV 4.
                System.loadLibrary("opencv_java4");
            }
        }

        private const val TAG_MAIN = "MainActivity"
        private const val TAG_MEDIAPIPE = "Mediapipe"
    }
}