import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class PointAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  0, or a positive number.
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to 0
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        PS: Tensor is a data type
        """
        if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')

        # Store the x,y coordinate of the points. (3 dimensional array slicing)
        points_xy = points[:, :2]
        # Store the strides of the points.
        points_stride = points[:, 2]
        # Store the lvl of each point
        points_lvl = torch.log2(
            points_stride).int()  # [3...,4...,5...,6...,7...] Store the point level. The strides are exponents of 2 incrementing at every level.
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max() #Store the maximum and minimum level
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]   # Store number of gt_boxes and points

        # assign gt box by adding the x1,x2 and y1,y2 (dividing by 2 for midpoint). Stores the midpoint of the gt_boxes.
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

        # Stores the width and height of the boxes (x2-x1) (y2-y1)
        # Clips all the elements in input with a minimum range and results in a Tensor
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)

        scale = self.scale

        # !! How does it scale?
        # Clamps the ground truth box levels for the backbone between minimum and maximum
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt of each point as 0 -> negative sample
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt distance (to this point) as infinite (to find the nearest point later)
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl # Stores a boolean array with True(1) where the values are equal for gt_lvl and points_lvl, False(0) otherwise
            points_index = points_range[lvl_idx] # points_index contains the values in point_range where indexes in lvl_indexes are True(1)

            # get the points in this level as is true wherever points_lvl == gt_lvl.
            # points_xy only contains the x,y corrdinates of the points.
            lvl_points = points_xy[lvl_idx, :]

            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]

            # compute the distance between gt center and
            #   all points in this level
            # !! Why do we divide by gt_wh?
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)

            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False)

            # the index of nearest k points to gt center in this level
            # !! Isn't min_dist_index = min_dist_point_index
            min_dist_points_index = points_index[min_dist_index]

            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point.

            """ less_than_recorded_index stores the new min_dist_points_index """
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]

            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index]

            # assign the result
            #!! Why idx + 1? 
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index]

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_points, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()    #Store the indexes where the assigned point is positive
            if pos_inds.numel() > 0:                        # If total number of elements in pos_inds > 0
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1] # Assign labels
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
