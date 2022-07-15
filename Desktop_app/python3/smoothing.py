from math import sqrt
from functools import reduce

TRIANGLE_W = 0x1
GAUSS_W = 0x2

w_makers = {TRIANGLE_W: lambda record_hist: range(1, record_hist + 1)}

fixation_thresh_min_mm = 10
fixation_thresh_max_mm = 20


class Smoothie:

    def __init__(self, record_hist, weight_type=TRIANGLE_W):

        self.weights = w_makers[weight_type](record_hist)
        self.weights = [w / float(sum(self.weights)) for w in self.weights]
        self.gaze_records = [[(0, 0)] * record_hist, [(0, 0)] * record_hist]

    def remove_wild_pts(self, gaze_pts):
        """ Detects if gaze is fixed then removes inaccurate gaze points
        """
        weighted_gaze_distance = {}
        is_gaze_fixed = False

        # determine if gazed is fixed by comparing current gaze points to their records history
        for i, gaze_pt in enumerate(gaze_pts):
            if gaze_pt is not None:
                gaze_dists = [sqrt((x - gaze_pt[0]) ** 2 + (y - gaze_pt[1]) ** 2) for (x, y) in self.gaze_records[i]]
                zipped = zip(self.weights, gaze_dists)
                weighted_gaze_distance[gaze_pt] = reduce(lambda w, d: (1, (w[0] * d[0] + w[1] * d[1])), zipped)[1]
                is_gaze_fixed = (weighted_gaze_distance[gaze_pt] < fixation_thresh_min_mm)

        # Filter out gaze points that differ wildly from their records histories if gaze is fixed
        if is_gaze_fixed:
            return [None if (g is None or weighted_gaze_distance[g] > fixation_thresh_max_mm) else g for g in gaze_pts]
        else:
            return gaze_pts

    def update_records(self, gaze_pts):

        for i, gaze_pt in enumerate(gaze_pts):
            if gaze_pt is not None:
                self.gaze_records[i].append(gaze_pt)
                self.gaze_records[i].pop(0)
                # self.gaze_records[i] = self.gaze_records[i]
                # self.gaze_records[i] = self.gaze_records[i][1:]

    def make_gaze_smooth(self, gaze_pts):
        """
        args: gaze pts mm
        """
        gaze_pts = self.remove_wild_pts(gaze_pts)
        self.update_records(gaze_pts)

        smoothed_pts = []
        for i in range(2):  # Find smoothed gaze for each eye
            zipped = zip(self.weights, self.gaze_records[i])  # has form [(1,(x,y)),(2,(x,y)),(3,(x,y))...]
            smoothed_pts.append(reduce(lambda w1, w2:
                                 (1, (w1[0] * w1[1][0] + w2[0] * w2[1][0], w1[0] * w1[1][1] + w2[0] * w2[1][1])),
                                 zipped)[1])  # only want last part of (w,(x,y))

        xs, ys = [x for (x, _) in smoothed_pts], [y for (_, y) in smoothed_pts]
        return sum(xs) / len(xs), sum(ys) / len(ys)  # return averaged gaze point
