//
//  ViewController.swift
//  CatVsDogClassifierSample
//
//  Created by Prianka Kariat on 17/06/19.
//  Copyright © 2019 Y Media Labs. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
  
  
  // MARK: Storyboards Connections
  @IBOutlet weak var collectionView: UICollectionView!
  
  // MARK: Instance Variables
  let imageNames: [String] = ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
  var inferences: [String] = ["","","","","",""]
 
  
  // Handles all data preprocessing and makes calls to run inference through the `Interpreter`.
  private var modelDataHandler: ModelDataHandler? =
    ModelDataHandler(modelFileInfo: MobileNet.modelInfo)
  
 
  override func viewDidLoad() {
    super.viewDidLoad()
    collectionView.dataSource = self
    collectionView.delegate = self
    collectionView.reloadData()
    
    // Do any additional setup after loading the view, typically from a nib.

    guard modelDataHandler != nil else {
      fatalError("Model set up failed")
    }

  }

}

// MARK: Extensions
extension ViewController: UICollectionViewDelegate, UICollectionViewDataSource {
  
  func numberOfSections(in collectionView: UICollectionView) -> Int {
    return 1
  }
  
  func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
    
    return imageNames.count
  }
  
  func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
    
    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "IMAGE_CELL", for: indexPath) as! ImageCell
    
    cell.imageView.image = UIImage(named: imageNames[indexPath.item])
    cell.inferenceLabel.text = inferences[indexPath.item]
    
    return cell
    
  }
  
  func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
    
    // Gets the pixel buffer from UIImage
    guard  let image = UIImage(named: imageNames[indexPath.item]),
           let pixelBuffer = image.pixelBuffer() else {
            return

  }
    
    // Hands over the pixel buffer to ModelDatahandler to perform inference
    let inferencesResults = modelDataHandler?.runModel(onFrame: pixelBuffer)
    
    // Formats inferences and resturns the results
    guard let firstInference = inferencesResults?.first else {
      return
    }
    
    inferences[indexPath.item] = firstInference.label
    collectionView.reloadData()

  }
  

  
}

