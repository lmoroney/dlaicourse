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

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import java.util.*

/** A simple View providing a render callback to other classes.  */
class OverlayView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private val callbacks = LinkedList<DrawCallback>()

    fun addCallback(callback: DrawCallback) {
        callbacks.add(callback)
    }

    @SuppressLint("MissingSuperCall")
    @Synchronized
    override fun draw(canvas: Canvas) {
        for (callback in callbacks) {
            callback.drawCallback(canvas)
        }
    }

    /** Interface defining the callback for client classes.  */
    interface DrawCallback {
        fun drawCallback(canvas: Canvas)
    }
}
