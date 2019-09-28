from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """ Base class for the assigner.
    """
    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
