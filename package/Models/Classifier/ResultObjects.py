def precision(tp, fp, _):
    if tp == 0:
        return 0
    return tp / (tp + fp)


def p(ct):
    tp, fp, fn, n = ct
    return precision(tp, fp, fn)


def recall(tp, _, fn):
    if tp == 0:
        return 0
    return tp / (tp + fn)


def r(ct):
    tp, fp, fn, n = ct
    return recall(tp, fp, fn)


def f1_metric(category_tuple):
    pv = p(category_tuple)
    rv = r(category_tuple)
    if pv + rv == 0:
        return 0
    return (2 * pv * rv) / (pv + rv)


class F1Object:
    def __init__(self, result):
        self.obj: list = result

    # Equivalent to sklearn - None
    def labels_score(self) -> list[float]:
        return [f1_metric(cat_tup) for cat_tup in self.obj]

    def label_score(self, pos_label: int) -> float:
        return self.obj[pos_label]

    # Equivalent to sklearn - micro
    def global_score(self) -> float:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        # cat_tuple = (true_pos, false_pos, false_neg)
        for cat_tuple in self.obj:
            tp, fp, fn, _ = cat_tuple
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        return f1_metric((true_positives, false_positives, false_negatives, _))

    # Equivalent to sklearn - macro
    def average_score(self):
        res = self.labels_score()
        return sum(res) / len(res)

    # Equivalent to sklearn - weighted
    def weighted_score(self):
        total = sum([n for _, _, _, n in self.obj])
        return [(n / total) * f1_metric((tp, fp, fn, n)) for tp, fp, fn, n in self.obj]

    def __str__(self):
        return ([f"#{n}-({tp}, {fp}, {fn}), "
                 for tp, fp, fn, n in self.obj]
                .__str__())


class Result:
    def __init__(self, num_test, num_success, f1):
        self.num_test: int = num_test
        self.num_success: int = num_success
        self.f1: F1Object = f1

    def accuracy(self):
        return self.num_success / self.num_test
