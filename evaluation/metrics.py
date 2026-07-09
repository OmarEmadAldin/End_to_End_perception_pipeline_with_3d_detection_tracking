import numpy as np
from scipy.optimize import linear_sum_assignment


MATCH_THRESHOLD_M = 2.0   # metres — match if within this distance


class MOTMetrics:

    def __init__(self, threshold_m=MATCH_THRESHOLD_M):
        self.threshold = threshold_m
        self.reset()

    def reset(self):
        self.total_gt     = 0
        self.total_tp     = 0
        self.total_fp     = 0
        self.total_fn     = 0
        self.total_ids    = 0
        self.total_dist   = 0.0
        self.total_matches= 0
        # track last-frame gt_id→pred_id mapping for ID switch detection
        self._prev_match  = {}   # gt_id → pred_id

    def add_frame(self, gt_objects, pred_objects):
      
        ng = len(gt_objects)
        np_ = len(pred_objects)
        self.total_gt += ng

        if ng == 0 and np_ == 0:
            return
        if ng == 0:
            self.total_fp += np_
            return
        if np_ == 0:
            self.total_fn += ng
            return

        # Distance matrix
        C = np.full((ng, np_), np.inf)
        for gi, g in enumerate(gt_objects):
            for pi, p in enumerate(pred_objects):
                d = np.hypot(g['x'] - p['x'], g['y'] - p['y'])
                if d < self.threshold:
                    C[gi, pi] = d

        # Hungarian assignment
        matched_g, matched_p = set(), set()
        tp, dist_sum, ids = 0, 0.0, 0

        if not np.all(np.isinf(C)):
            C_f = np.where(np.isinf(C), self.threshold * 10, C)
            ri, ci = linear_sum_assignment(C_f)
            for g_idx, p_idx in zip(ri, ci):
                if C[g_idx, p_idx] < self.threshold:
                    tp       += 1
                    dist_sum += C[g_idx, p_idx]
                    matched_g.add(g_idx)
                    matched_p.add(p_idx)

                    g_id = gt_objects[g_idx]['id']
                    p_id = pred_objects[p_idx]['id']
                    # ID switch: this GT was matched to a DIFFERENT prediction before
                    if g_id in self._prev_match and self._prev_match[g_id] != p_id:
                        ids += 1
                    self._prev_match[g_id] = p_id

        fp = np_ - len(matched_p)
        fn = ng  - len(matched_g)

        self.total_tp     += tp
        self.total_fp     += fp
        self.total_fn     += fn
        self.total_ids    += ids
        self.total_dist   += dist_sum
        self.total_matches+= tp

    def compute(self):
       
        gt = max(self.total_gt, 1)   # avoid div/0

        mota = 1.0 - (self.total_fn + self.total_fp + self.total_ids) / gt

        motp = (self.total_dist / self.total_matches
                if self.total_matches > 0 else float('inf'))

        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn

        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1        = (2 * precision * recall / max(precision + recall, 1e-9))

        return {
            'MOTA'     : round(mota, 4),
            'MOTP_m'   : round(motp, 4),
            'IDS'      : self.total_ids,
            'TP'       : tp,
            'FP'       : fp,
            'FN'       : fn,
            'Recall'   : round(recall, 4),
            'Precision': round(precision, 4),
            'F1'       : round(f1, 4),
            'Total_GT' : self.total_gt,
        }
