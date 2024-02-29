def ensemble_net(image):
  """
  This function takes an image as input and outputs bounding boxes for detected classes along with their objectness score.

  Args:
    image: The input image.

  Returns:
    A list of bounding boxes for detected classes along with their objectness score.
  """

    # Set the IoU threshold for ensembling.
  threshold = 0.5

  # Get detections from Faster R-CNN and YOLOv5 models.
  efficient_detections = get_efficient_detections(image)
  yolov8_detections = get_yolov8_detections(image)

  # Initialize empty list to store ensembled detections.
  ensembled_detections = []

  # Loop through each efficientDet detection.
  for efficient_detection in efficient_detections:
    efficient_bbox, efficient_class, efficient_score = efficient_detection

    # Find the corresponding YOLOv8 detection for the same class.
    corresponding_yolov8_detection = None
    for yolov8_detection in yolov8_detections:
      yolov8_bbox, yolov8_class, yolov8_score = yolov8_detection
      if yolov8_class == efficient_class:
        corresponding_yolov8_detection = yolov8_detection
        break

    # If there is a corresponding YOLOv8 detection.
    if corresponding_yolov8_detection:
      yolov8_bbox, yolov8_class, yolov8_score = corresponding_yolov8_detection

      # Check if the bounding boxes are close enough (IoU > threshold).
      if is_iou_greater_than_threshold(efficient_bbox, yolov8_bbox, threshold):
        # If the efficientDet score is higher, use the efficientDet detection.
        if efficient_score > yolov8_score:
          ensembled_detections.append([efficient_bbox, efficient_class, efficient_score])
        # Else, use the YOLOv8 detection.
        else:
          ensembled_detections.append([yolov8_bbox, yolov8_class, yolov8_score])
      # Else, if the bounding boxes are not close enough, use the detection with the higher score.
      else:
        if efficient_score > yolov8_score:
          ensembled_detections.append([efficient_bbox, efficient_class, efficient_score])
        else:
          ensembled_detections.append([yolov8_bbox, yolov8_class, yolov8_score])
    # Else, if there is no corresponding YOLOv8 detection, use the EfficientDet detection.
    else:
      ensembled_detections.append([efficient_bbox, efficient_class, efficient_score])

  return ensembled_detections

