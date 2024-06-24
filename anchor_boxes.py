import numpy as np

class AnchorBoxes:
    """
    Generate anchor boxes for object detection.

    Args:
        namespace_config (Namespace): A namespace containing the configuration settings.
        img_size (tuple): Tuple containing the height and width of the input image.

    Args from NamespaceConfig:
        anchor_boxes (Namespace): Namespace containing anchor box configuration.
        anchor_boxes.scales (list): List of scales for the anchor boxes.
        anchor_boxes.ratios (list): List of aspect ratios for the anchor boxes.
        anchor_boxes.num (int): Number of anchor boxes to generate per scale and ratio.

    Attributes:
        lower_threshold (float): Lower threshold for IoU.
        upper_threshold (float): Upper threshold for IoU.
        anchor_boxes (numpy.ndarray): Anchor boxes of shape (n, 4).
    """

    def __init__(self, namespace_config, img_size):
        self.scales = namespace_config.anchor_boxes.scales
        self.ratios = namespace_config.anchor_boxes.ratios
        self.num_anchor_boxes = namespace_config.anchor_boxes.num
        self.img_size = img_size

        self.lower_threshold = 0.2
        self.upper_threshold = 0.3

        self.anchor_boxes = self.generate_anchor_boxes(self.num_anchor_boxes, self.scales, self.ratios, self.img_size)

    def generate_anchor_boxes(self, n, scales, ratios, img_size):
        """
        Generate anchor boxes for object detection.
        
        Args:
            n (int): Number of anchors to generate per scale and ratio.
            scales (list): List of scales for the anchor boxes.
            ratios (list): List of aspect ratios for the anchor boxes.
            img_size (tuple): Tuple containing the height and width of the input image.
        
        Returns:
            numpy.ndarray: An array of shape (n * len(scales) * len(ratios), 4)
                        containing the anchor boxes in (x1, y1, x2, y2) format.
        """
        h = img_size[0]
        w = img_size[1]

        center_x = np.linspace(0, w, n + 1)[:-1] + w / (2 * n)
        center_y = np.linspace(0, h, n + 1)[:-1] + h / (2 * n)
        cx, cy = np.meshgrid(center_x, center_y)
        
        centers = np.vstack([cx.ravel(), cy.ravel()]).transpose()
        
        anchor_boxes = []
        
        for (c_x, c_y) in centers:
            for scale in scales:
                for ratio in ratios:
                    half_w = scale * np.sqrt(ratio) / 2
                    half_h = scale / np.sqrt(ratio) / 2
                    
                    x1 = max(c_x - half_w, 0)
                    y1 = max(c_y - half_h, 0)
                    x2 = min(c_x + half_w, w)
                    y2 = min(c_y + half_h, h)
                    
                    anchor_boxes.append([x1, y1, x2, y2]) 
        anchor_boxes = np.array(anchor_boxes)
        return anchor_boxes
    
    def IoU(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list or numpy.ndarray): Coordinates of the first bounding box in format [x1, y1, x2, y2].
            box2 (list or numpy.ndarray): Coordinates of the second bounding box in format [x1, y1, x2, y2].

        Returns:
            float: Intersection over Union (IoU) score.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
        
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area

        return iou
    
    def assign_anchor_boxes(self, true_boxes, anchor_boxes, lower_threshold, upper_threshold):
        """
        Assign each anchor box to a ground truth box.
        
        Args:
            true_boxes (numpy.ndarray): Ground truth bounding boxes of shape (n, 4).
            anchor_boxes (numpy.ndarray): Anchor boxes of shape (m, 4).
            lower_threshold (float): Lower threshold for IoU.
            upper_threshold (float): Upper threshold for IoU.

        Returns:
            numpy.ndarray: An array of shape (n, 1) containing the index of the assigned ground truth box for each anchor box.
        """
        
        m = anchor_boxes.shape[0]
        n = true_boxes.shape[0]
        
        iou_matrix = np.zeros((m, n))
        
        for i in range(m):
            for j in range(n):
                iou_matrix[i, j] = self.IoU(anchor_boxes[i], true_boxes[j])
        assigned_gt_idx = np.full(m, -1)
        iou_list = []
        for i in range(m):
            gt_idx = np.argmax(iou_matrix[i])
            iou = iou_matrix[i, gt_idx]
            iou_list.append(iou)
            if iou >= upper_threshold:
                assigned_gt_idx[i] = gt_idx
            elif iou < lower_threshold:
                assigned_gt_idx[i] = -1 # background
            else:
                assigned_gt_idx[i] = -2 # ignore

        return assigned_gt_idx
    
    def calculate_offset(self, true_boxes, anchor_boxes, assigned_boxes):
        """
        Calculate the offset values for anchor boxes.

        Args:
            anchor_boxes (numpy.ndarray): Anchor boxes of shape (n, 4).
            true_boxes (numpy.ndarray): Ground truth bounding boxes of shape (n, 4).
            assigned_boxes (numpy.ndarray): Index of the assigned ground truth box for each anchor box.

        Returns:
            numpy.ndarray: Offset values for xyxy of shape (n, 4).
        """
        
        offsets = np.zeros_like(anchor_boxes)

        for i, box in enumerate(anchor_boxes):
            if assigned_boxes[i] >= 0:
                gt_box = true_boxes[assigned_boxes[i]]
                offsets[i] = [(gt_box[0] - box[0]) / (box[2] - box[0]),
                            (gt_box[1] - box[1]) / (box[3] - box[1]),
                            (gt_box[2] - box[2]) / (box[2] - box[0]),
                            (gt_box[3] - box[3]) / (box[3] - box[1])]
        return offsets
    
    def assign_class_label(self, anchor_boxes, assigned_boxes, true_labels):
        """
        Assign class labels to anchor boxes.

        Args:
            anchor_boxes (numpy.ndarray): Anchor boxes of shape (n, 4).
            assigned_boxes (numpy.ndarray): Index of the assigned ground truth box for each anchor box.
            true_labels (numpy.ndarray): true labels for each ground truth box.

        Returns:
            numpy.ndarray: Class labels for each anchor box.
        """

        class_labels = np.full(anchor_boxes.shape[0], -1)
        for i, idx in enumerate(assigned_boxes):
            if idx >= 0:
                class_labels[i] = true_labels[idx]
            elif idx == -1:
                class_labels[i] = 11
            else:
                class_labels[i] = 12
        return class_labels
    
    def class_offset_split(self, true_boxes, true_classes):
        """
        Split the class labels and offset values.

        Args:
            true_boxes (list): List of ground truth bounding boxes.
            true_classes (list): List of class labels.

        Returns:
            tuple: A tuple containing the offset values and class labels.
        """
        offsets = []
        class_labels = []
        for img_dim, cls in zip(true_boxes, true_classes):
            assigned_boxes = self.assign_anchor_boxes(np.array(img_dim), self.anchor_boxes)
            offset = self.calculate_offset(self.anchor_boxes, np.array(img_dim), assigned_boxes)
            labels = self.assign_class_label(self.anchor_boxes, assigned_boxes, np.array(cls))
            offsets.append(offset)
            class_labels.append(labels)

        return offsets, class_labels