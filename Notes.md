## Tensorflow organization

 Let's start with a quick review of the design and architecture of TensorFlow.js, and you can see that here. The goal is twofold. First, we want to make it easy for you to code against it with a friendly high-level API, but you can also go lower into the APIs and program against them directly too. It's designed to run in the browser as well as an Node.js server. Up to this point, you've been programming convolutional neural networks and deep neural networks using the Keras API in TensorFlow. The layers API and TensorFlow.js looks and feels a lot like Keras. So what you've been learning, you'll be able to use albeit with a slightly different syntax due to using JavaScript instead of Python.

<img width=600px src="images/Tensorflow_api.jpg" />

The low-level APIs are called the core APIs. They're designed to work with a TensorFlow saved model formats, which in TensorFlow 2.0 is designed to be a standard file format which can be used across the Python APIs, the JavaScript once, and even TensorFlow Lite for mobile and embedded devices. The Core API then works with the browser and can take advantage of WebGL for accelerated training and inference. Also on Node.js, you can build server-side or terminal applications using it. These can then take advantage of CPUs, GPUs, and TPUs depending on what's available to your machine. We're going to build the simplest possible neural network. One which matches two sets of numbers to see that the relationship between them is y equals two x minus 1. We'll do that in JavaScript.

## Browser Setup

There are two main ways to get TensorFlow.js in your browser based projects:

- Using [script tags](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage).
- Installation from [NPM](https://www.npmjs.com/) and using a build tool like [Parcel](https://parceljs.org/), [WebPack](https://webpack.js.org/), or [Rollup](https://rollupjs.org/guide/en).

If you are new to web development, or have never heard of tools like webpack or parcel, *we recommend you use the script tag approach*. If you are more experienced or want to write larger programs it might be worthwhile to explore using build tools.

### Usage via Script Tag

Add the following script tag to your main HTML file.

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>

```javascript
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
  // Use the model to do inference on a data point the model hasn't seen before:
  model.predict(tf.tensor2d([5], [1, 1])).print();
  // Open the browser devtools to see the output
});
  
```

```javascript
const convertedData =
                  trainingData.map(({xs, ys}) => {
                      const labels = [
                            ys.species == "setosa" ? 1 : 0,  <!--Ask for setosa, if true then 1, else 0-->
                            ys.species == "virginica" ? 1 : 0,
                            ys.species == "versicolor" ? 1 : 0
                      ] 
                      return{ xs: Object.values(xs), ys: Object.values(labels)};
                  }).batch(10);
<!--This is to do One Hot encoding on a dictionary-->
```

