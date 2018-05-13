from imgaug import augmenters as iaa


def fast_seq(train_mode=False):
    if train_mode:
        fast_seq = iaa.SomeOf((1, 2),
                              [iaa.Fliplr(0.5),
                               iaa.Flipud(0.5),
                               iaa.Affine(rotate=(0, 360),
                                          translate_percent=(-0.1, 0.1)),
                               iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='reflect')
                               ], random_order=True)
    else:
        fast_seq = iaa.Noop()
    return fast_seq
