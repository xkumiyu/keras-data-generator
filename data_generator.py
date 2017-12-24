import pathlib

from keras.utils import to_categorical
import numpy as np
from PIL import Image


class ImageDataGenerator(object):
    def __init__(self, rescale=None):
        self.rescale = rescale
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, directory, classes, batch_size=32):
        classes = {v: i for i, v in enumerate(sorted(classes))}
        while True:
            for path in pathlib.Path(directory).iterdir():
                with Image.open(path) as f:
                    self.images.append(np.asarray(f.convert('RGB'), dtype=np.float32))
                _, y = path.stem.split('_')
                self.labels.append(to_categorical(classes[y], len(classes)))

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets
