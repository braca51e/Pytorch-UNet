# import the necessary packages
import torch

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	#if boxes.dtype == torch.int:
	boxes = boxes.float()

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	w = boxes[:, 2]
	h = boxes[:, 3]
	x2 = x1 + w
	y2 = y1 + h

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-right y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = torch.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:

		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last].item()
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = torch.max(x1[i], x1[idxs[:last]])
		yy1 = torch.max(y1[i], y1[idxs[:last]])
		xx2 = torch.min(x2[i], x2[idxs[:last]])
		yy2 = torch.min(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
		h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		condition = (overlap < overlapThresh)
		keep_list = overlap[condition].nonzero().squeeze()
        #delete last index (currently selected) from index list
		idxs = idxs[:-1]
        #delete indexes that don't satisfy the overlap condition
		idxs = torch.index_select(idxs, 0, keep_list)

	# return only the bounding boxes that were picked
	return boxes[pick].int()
