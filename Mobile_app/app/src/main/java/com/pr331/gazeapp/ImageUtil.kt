package com.pr331.gazeapp

import android.graphics.*
import android.os.Environment
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt


class ImageUtil {
    private val LOGINGTAG: String = "ImageUtil"


    fun getLandmarksDebugString(landmarks: LandmarkProto.NormalizedLandmarkList): String? {
        var landmarkIndex = 0
        var landmarksString = ""
        for (landmark in landmarks.landmarkList) {
            // landmarksString += """Landmark[$landmarkIndex]: (${landmark.x}, ${landmark.y}, ${landmark.z})"""
            landmarksString += """(${landmark.x}, ${landmark.y}, ${landmark.z})\n"""
            ++landmarkIndex
        }
        return landmarksString
    }


    fun saveBitmap2File(region: Bitmap, filename: String) {
        // Save image to
        var path = ""
        when {
            filename.contains("-Face") -> {
                path = "/sdcard/face"
            }
            filename.contains("-Eyes") -> {
                path = "/sdcard/eyes"
            }
            filename.contains("-LeftEye") -> {
                path = "/sdcard/leftEye"
            }
            filename.contains("-RightEye") -> {
                path = "/sdcard/rightEye"
            }
        }

        val file = File(path)
        try {
            val stream: OutputStream = FileOutputStream(file)
            region.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            stream.flush()
            stream.close()
            Log.i(LOGINGTAG, "Saved file ${file.absolutePath}")
        } catch (e: IOException) {
            e.printStackTrace()
            Log.i(LOGINGTAG, "Saved file error: ${e.toString()}")
        }
    }
    fun saveImage(finalBitmap: Bitmap, filename: String) {
        val root = Environment.getExternalStorageDirectory().toString()
        val myDir = File("$root/saved_images")
//        myDir.mkdirs()
        val file = File(myDir, filename)
        if (file.exists()) {
            file.delete()
            file.createNewFile()
        }
        try {
            val out = FileOutputStream(file)
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            out.flush()
            out.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }



    private fun getDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val ret = sqrt(((x1 - x2) * 1.0).pow(2.0) + ((y1 - y2) * 1.0).pow(2.0))
        return ret.toFloat()
    }

    fun getXYmm(coord: LandmarkProto.NormalizedLandmark, zDistanceMm: Float? = 0.0f, width: Int, height: Int): List<Float> {
        val dist2Center = getDistance(coord.x * width, coord.y * height, width / 2.0f, height / 2.0f)
        var useDistance = 10.0f
        if (zDistanceMm != null) {
            if (zDistanceMm > 0.0f) {
                useDistance = zDistanceMm
            }
        }
        val xMm = (coord.x * width - width / 2.0f) * (useDistance / dist2Center)
        val yMm = (coord.y * height - height / 2.0f) * (useDistance / dist2Center)

        return listOf(xMm, yMm)
    }

    fun getRect(contour: LandmarkProto.NormalizedLandmarkList, indexing: List<Int>, width: Int, height: Int): Rect {
        var xleft = 10000.0f
        var yleft = 10000.0f
        var xright = 0.0f
        var yright = 0.0f
        val landmarksList = contour.landmarkList

        for (index in indexing) {
            xleft = min(xleft, landmarksList[index].x * width)
            yleft = min(yleft, landmarksList[index].y * height)
            xright = max(xright, landmarksList[index].x * width)
            yright = max(yright, landmarksList[index].y * height)
        }
        val rect = Rect(xleft.toInt(), yleft.toInt(), xright.toInt(), yright.toInt())
        return rect
    }

    fun cropResizePad(bitmap: Bitmap, rect: Rect, target_width: Int, target_height: Int, grayscale: Boolean = true): Bitmap? {
        val width = rect.right - rect.left
        val height = rect.bottom - rect.top

        try {
            val croppedImage: Bitmap
            if (width > 0 && height > 0) {
                croppedImage = Bitmap.createBitmap(
                    bitmap,
                    rect.left,
                    rect.top,
                    width,
                    height
                )
            } else {
                croppedImage =
                    Bitmap.createBitmap(target_width, target_height, Bitmap.Config.ARGB_8888)
            }
            val resizedImage = Bitmap.createScaledBitmap(
                croppedImage,
                target_width,
                target_height,
                false
            )

            if (grayscale) {
                // Convert croppedImage to grayscale
                val imgGrayScale = Bitmap.createBitmap(
                    resizedImage.width,
                    resizedImage.height,
                    Bitmap.Config.ARGB_8888
                )
                val c = Canvas(imgGrayScale)
                val paint = Paint()
                val cm = ColorMatrix()
                cm.setSaturation(0.0f)
                val f = ColorMatrixColorFilter(cm)
                paint.setColorFilter(f)
                c.drawBitmap(resizedImage, 0.0f, 0.0f, paint)
                return imgGrayScale
            } else {
                return resizedImage
            }
        } catch (e: IllegalArgumentException) {
            return null
        }
    }

    fun getMean(list_point: MutableList<DoubleArray>, indexing: List<Int>, default_dim: Int = 4): DoubleArray {
        val retMean = DoubleArray(default_dim) { 0.0 }
        for (idx in indexing) { // [maybe list_point size is 18 ->> indexing = [13,14,15,16,17]
            for (ddim in 0 until default_dim) {
                retMean[ddim] += list_point[idx][ddim] / indexing.size
            }
        }

        return retMean
    }
}