class RPSDataset {

  constructor() {
    // When we initialize the class the labels will
    // initialize in an empty array
    this.labels = []
  }

  addExample(example, label) {
      // the example is the output from truncated mobilenet
      // the labels are 0,1 or 2 from the images
    if (this.xs == null) {
        // for the first sample the xs is null
        // set xs to tf.keep(example)
        // tf.keep tells that we want to keep the tensor so it will not
        // be discarded by tf.tidy()
      this.xs = tf.keep(example);
      this.labels.push(label);
    } else {
        // for subsequence samples just append the new one
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
        // then dispose it
      oldX.dispose();
    }
  }
  
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
