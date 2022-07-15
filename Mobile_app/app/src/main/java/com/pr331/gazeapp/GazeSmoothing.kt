package com.pr331.gazeapp

//fixme: The gaze smoothing seems to not work
internal class GazeSmoothing(recordHistory: Int) {
    private val weights: MutableList<Double> = ArrayList()

    var recordHistory = MutableList(2, {MutableList(recordHistory, {MutableList(2, {0.0})})})
    var fixationThresholdMin = 10.0
    var fixationThresholdMax = 20.0

    /*
    @ param: List<List<Double>>     # A 2D list which has a form [[a,b], [c,d]], for the left and right eye
     */
    private fun removeWildPoints(gazePoints: List<List<Double>?>): List<List<Double>?>? {
        /*
    Detects if gaze if fixed then removes inaccurate gaze points
     */
        val weightedGazeDistance: MutableMap<List<Double>?, Double> = HashMap()
        var isGazeFixed = false
        for (i in gazePoints.indices) {
            val gazePoint = gazePoints[i]
            if (gazePoint != null) {
                val gazeDistance: MutableList<Double> = ArrayList()
                for (j in 0..2) {
                    val x2 = recordHistory[i][j][0]
                    val y2 = recordHistory[i][j][1]
                    val xGazePoint = gazePoint[0]
                    val yGazePoint = gazePoint[1]
                    val gDistance = Math.sqrt(Math.pow(x2 - xGazePoint, 2.0) + Math.pow(y2 - yGazePoint, 2.0))
                    gazeDistance.add(j, gDistance)
                }
                val weightedDist = weightedDistance(weights, gazeDistance)
                weightedGazeDistance[gazePoint] = weightedDist
                isGazeFixed =
                        weightedGazeDistance[gazePoint]!! < fixationThresholdMin
            }
        }
        val gazept: MutableList<List<Double>?> = ArrayList()
        return if (isGazeFixed) {
            for (gazePoint in gazePoints) {
                if (gazePoint != null || weightedGazeDistance[gazePoint]!! < fixationThresholdMax) {
                    gazept.add(gazePoint)
                } else {
                    return null
                }
            }
            gazept
        } else {
            gazePoints
        }
    }

    fun updateRecordHistory(gazePoints: List<List<Double>?>?) {
        if (gazePoints != null) {
            val gazePoint1 = gazePoints[0]
            val gazePoint2 = gazePoints[1]
            recordHistory[0].add(gazePoint1 as MutableList<Double>)
            recordHistory[0].removeFirst()
            recordHistory[1].add(gazePoint2 as MutableList<Double>)
            recordHistory[1].removeFirst()

        }
    }

    fun makeGazeSmooth(gazePoints: List<List<Double>?>): List<Double> {
        // so far the removeWildPoints is neutral
        val gazepts = removeWildPoints(gazePoints)
        updateRecordHistory(gazepts)
        val smoothedPts: MutableList<List<Double>> = ArrayList()
        for (i in 0..1) {
            val singleEyeGazePt = smoothPoints(weights, recordHistory[i])
            smoothedPts.add(singleEyeGazePt)
        }
        val xs: MutableList<Double> = ArrayList()
        val ys: MutableList<Double> = ArrayList()
        xs.add(0, smoothedPts[0][0])
        xs.add(1, smoothedPts[1][0])
        ys.add(0, smoothedPts[0][1])
        ys.add(1, smoothedPts[1][1])
        val xAve = (xs[0] + xs[1]) / 2
        val yAve = (ys[0] + ys[1]) / 2
        val retList: MutableList<Double> = ArrayList()
        retList.add(xAve)
        retList.add(yAve)
        return retList
    }

    fun weightedDistance(weights: List<Double>, distance: List<Double>): Double {
        val tempList = doubleArrayOf(1.0, 1.0)
        for (i in distance.indices) {
            tempList[1] = tempList[0] * weights[i] + tempList[1] * distance[i]
        }
        return tempList[1]
    }

    fun smoothPoints(weights: List<Double>, points: List<MutableList<Double>>): List<Double> {
        val tempListV2: MutableList<MutableList<Double>> = ArrayList()
        val tempList = arrayOf(
                doubleArrayOf(1.0), doubleArrayOf(
                weights[0],
                weights[0]
        )
        )
        val firstElement: MutableList<Double> = ArrayList()
        firstElement.add(0, 1.0)
        val secondElement: MutableList<Double> = ArrayList()
        secondElement.add(0, weights[0])
        secondElement.add(1, weights[0])
        tempListV2.add(0, firstElement)
        tempListV2.add(1, secondElement)
        for (i in weights.indices) {
            val value1 = tempListV2[0][0] * tempListV2[1][0] + weights[i] * points[i][0]
            val value2 = tempListV2[0][0] * tempListV2[1][1] + weights[i] * points[i][1]
            tempListV2[1][0] = value1
            tempListV2[1][1] = value2
        }
        return tempListV2[1]
    }

    private fun sumOfRange(rangeEnd: Int): Double {
        var sum = 0.0
        for (i in 1 until rangeEnd) {
            sum = sum + i
        }
        return sum
    }

    init {
        val sumOfRecordsHistory = sumOfRange(recordHistory + 1)
        for (i in 1 until recordHistory + 1) {
            val w = i / sumOfRecordsHistory
            weights.add(i - 1, w)
        }
    }
}

//////////////////////////////////////////////////
//            Driver program                    //
//////////////////////////////////////////////////
fun main(){
    val gazeSmoothing = GazeSmoothing(6)
    val gazePointsInMM = mutableListOf(
            mutableListOf(mutableListOf(400.0,200.0), mutableListOf(390.0, 190.0)),
            mutableListOf(mutableListOf(900.0, 1000.0), mutableListOf(800.0, 1100.0)))
    for (i in 0..1) {
        val xyGaze = gazeSmoothing.makeGazeSmooth(gazePointsInMM[i])
        println("gaze screen value x:" + xyGaze[0] + " y:" + xyGaze[1])
    }
    val recordHistory2Output: List<List<List<Double>>> = gazeSmoothing.recordHistory
    println("recordHistory2Output: $recordHistory2Output")
}