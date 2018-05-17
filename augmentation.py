from imgaug import augmenters as iaa
import numpy as np

import pipeline_config as cfg


def fast_seq(with_augmentation):
    if with_augmentation:
        aug = iaa.Sequential([iaa.SomeOf((1, 2),
                                         [iaa.Fliplr(0.5),
                                          iaa.Flipud(0.5),
                                          iaa.Affine(rotate=(0, 360),
                                                     translate_percent=(-0.1, 0.1)),
                                          iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='reflect')
                                          ], random_order=True),
                              iaa.Scale({'height': cfg.IMAGE_PARAMS.h, 'width': cfg.IMAGE_PARAMS.w}, deterministic=True)
                              ])
    else:
        aug = iaa.Scale({'height': cfg.IMAGE_PARAMS.h, 'width': cfg.IMAGE_PARAMS.w}, deterministic=True)
    return aug


def fast_augmentation(image, with_augmentation):
    seq = fast_seq(with_augmentation)
    image = image.astype(np.uint8)

    augmented = seq.augment_image(image).astype(np.float64)
    augmented /= 255.
    return augmented
