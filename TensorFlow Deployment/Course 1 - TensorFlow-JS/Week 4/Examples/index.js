let mobilenet;
let model;
// Const webcam object stored in webcam.js
// webcam.js is provided in th download
const webcam = new Webcam(document.getElementById('wc'));
// declare dataset
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0;
let isPredicting = false;

async function loadMobilenet() {
    // get mobile net model from hosted url
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    // getting a layer named conv_pw_13_relu
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
    // for One Hot encoded
  dataset.encodeLabels(4);
    // the input for the model will be the output from mobilenet
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 4, activation: 'softmax'})
    ]
  });
    
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
        case "3":
			spockSamples++;
			document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
    // The image captured is not introduced in the dataset
    // The dataset will contain the prediction from the 
    // mobilenet of the image and the label
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
      
      // read frame from webcam, get prediciton
      // from mobilenet (truncated) and then use it to get a 
      // prediction from the model. Return it as 1D
      // Tensor and use argmax to get 0,1 or 2 to return it as
      // classification (the prediction is one hot encoded)
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
      
      // Await for the result and go to aswitch case
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
        case 3:
			predictionText = "I see Spock";
			break;
	}
      // populate the prediction div in the html with the text
	document.getElementById("prediction").innerText = predictionText;
			
    // dispose the class
    predictedClass.dispose();
    // go to the next frame so the page keeps responsive
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
