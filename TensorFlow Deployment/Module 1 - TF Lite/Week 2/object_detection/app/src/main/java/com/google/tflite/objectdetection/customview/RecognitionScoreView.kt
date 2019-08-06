/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.tflite.objectdetection.customview

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import com.google.tflite.objectdetection.tflite.Classifier.Recognition

class RecognitionScoreView(context: Context, set: AttributeSet) : View(context, set), ResultsView {
    private val textSizePx: Float
    private val fgPaint: Paint
    private val bgPaint: Paint
    private var results: List<Recognition>? = null

    init {

        textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        fgPaint = Paint()
        fgPaint.textSize = textSizePx

        bgPaint = Paint()
        bgPaint.color = -0x33bd7a0c
    }

    override fun setResults(results: List<Recognition>) {
        this.results = results
        postInvalidate()
    }

    public override fun onDraw(canvas: Canvas) {
        val x = 10
        var y = (fgPaint.textSize * 1.5f).toInt()

        canvas.drawPaint(bgPaint)

        if (results != null) {
            for (recog in results!!) {
                canvas.drawText(recog.title + ": " + recog.confidence, x.toFloat(), y.toFloat(), fgPaint)
                y += (fgPaint.textSize * 1.5f).toInt()
            }
        }
    }

    companion object {
        private val TEXT_SIZE_DIP = 14f
    }
}
