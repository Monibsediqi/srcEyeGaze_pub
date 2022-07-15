package com.pr331.gazeapp

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log

class GazeGraphic constructor(overlay: GraphicOverlay?, private val drawPoint: List<Float>?): GraphicOverlay.Graphic(overlay) {
    private val gazePointPaint: Paint

    init {
        val selectedColor = Color.WHITE

        val gazePointColor = Color.RED
        gazePointPaint = Paint()
        gazePointPaint.color = gazePointColor
    }

    /** Draws the face annotations for position on the supplied canvas.  */
    override fun draw(canvas: Canvas) {
        // Draws a circle at the position of the detected face, with the face's track id below.

        if (drawPoint != null) {
            // Draws a circle at the position of the detected face, with the face's track id below.
            canvas.drawCircle(drawPoint[0], drawPoint[1], GAZE_POSITION_RADIUS, gazePointPaint)
            Log.v("Gaze graphic", "Draw gaze point")
        }
    }

    companion object {
        private const val GAZE_POSITION_RADIUS = 4.0f
        private const val ID_TEXT_SIZE = 30.0f
        private const val ID_Y_OFFSET = 40.0f
        private const val BOX_STROKE_WIDTH = 5.0f
        private const val NUM_COLORS = 10
        private val COLORS =
            arrayOf(
                intArrayOf(Color.BLACK, Color.WHITE),
                intArrayOf(Color.WHITE, Color.MAGENTA),
                intArrayOf(Color.BLACK, Color.LTGRAY),
                intArrayOf(Color.WHITE, Color.RED),
                intArrayOf(Color.WHITE, Color.BLUE),
                intArrayOf(Color.WHITE, Color.DKGRAY),
                intArrayOf(Color.BLACK, Color.CYAN),
                intArrayOf(Color.BLACK, Color.YELLOW),
                intArrayOf(Color.WHITE, Color.BLACK),
                intArrayOf(Color.BLACK, Color.GREEN)
            )
    }
}