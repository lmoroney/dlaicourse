package com.google.tflite.imageclassification.sample.tflite

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.experimental.and


class Classifier(assetManager: AssetManager, modelPath: String, labelPath: String, inputSize: Int) {
    private var INTERPRETER: Interpreter
    private var LABEL_LIST: List<String>
    private val INPUT_SIZE: Int = inputSize
    private val PIXEL_SIZE: Int = 3
    private val IMAGE_MEAN = 0
    private val IMAGE_STD = 255.0f
    private val MAX_RESULTS = 3
    private val THRESHOLD = 0.4f

    data class Recognition(
            var id: String = "",
            var title: String = "",
            var confidence: Float = 0F
    ) {
        override fun toString(): String {
            return "Title = $title, Confidence = $confidence)"
        }
    }

    init {
        val tfliteOptions = Interpreter.Options()
        tfliteOptions.setNumThreads(5)
        tfliteOptions.setUseNNAPI(true)
        INTERPRETER = Interpreter(loadModelFile(assetManager, modelPath),tfliteOptions)
        LABEL_LIST = loadLabelList(assetManager, labelPath)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }

    }

    fun recognizeImage(bitmap: Bitmap): List<Recognition> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) { ByteArray(LABEL_LIST.size) }
        INTERPRETER.run(byteBuffer, result)
        return getSortedResult(result)
    }


    private fun addPixelValue(byteBuffer: ByteBuffer, intValue: Int): ByteBuffer {

        byteBuffer.put((intValue.shr(16) and 0xFF).toByte())
        byteBuffer.put((intValue.shr(8) and 0xFF).toByte())
        byteBuffer.put((intValue and 0xFF).toByte())
        return byteBuffer
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
        imgData.order(ByteOrder.nativeOrder())
        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)


        imgData.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        var pixel = 0
        val startTime = SystemClock.uptimeMillis()
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val `val` = intValues[pixel++]
                addPixelValue(imgData, `val`)
            }
        }
        return imgData;
//        val endTime = SystemClock.uptimeMillis()
//        LOGGER.v("Timecost to put values into ByteBuffer: " + (endTime - startTime))
    }


    private fun getSortedResult(labelProbArray: Array<ByteArray>): List<Recognition> {
        Log.d("Classifier", "List Size:(%d, %d, %d)".format(labelProbArray.size, labelProbArray[0].size, LABEL_LIST.size))

        val pq = PriorityQueue(
                MAX_RESULTS,
                Comparator<Recognition> { (_, _, confidence1), (_, _, confidence2)
                    ->
                    java.lang.Float.compare(confidence1, confidence2) * -1
                })

        for (i in LABEL_LIST.indices) {
            val confidence = labelProbArray[0][i]
            if (confidence >= THRESHOLD) {
                Log.d("confidence value:", "" + confidence);
                pq.add(Recognition("" + i,
                        if (LABEL_LIST.size > i) LABEL_LIST[i] else "Unknown",
                        ((confidence).toFloat() / 255.0f)
                ))
            }
        }
        Log.d("Classifier", "pqsize:(%d)".format(pq.size))

        val recognitions = ArrayList<Recognition>()
        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionsSize) {
            recognitions.add(pq.poll())
        }
        return recognitions
    }

}