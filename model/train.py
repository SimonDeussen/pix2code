#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import tensorflow as tf

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth = True
config.log_device_placement=True

sess = tf.Session(config=config)

import sys

from classes.dataset.Generator import *
from classes.model.pix2code import *

import os

def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    model = pix2code(input_shape, output_size, output_path)
    

    print( output_path + "model_summary" )
    model_summary = open(output_path + "model_summary" + ".txt", "w+")
    model.model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    model_summary.write("\n")
    model_summary.write("CONTEXT_LENGTH " + str(CONTEXT_LENGTH) + '\n')
    model_summary.write("IMAGE_SIZE " + str(IMAGE_SIZE) + '\n')
    model_summary.write("BATCH_SIZE " + str(BATCH_SIZE) + '\n')
    model_summary.write("EPOCHS " + str(EPOCHS) + '\n')
    model_summary.write("STEPS_PER_EPOCH " + str(STEPS_PER_EPOCH) + '\n')
    model_summary.write("input_shape " + str(input_shape) + '\n')
    model_summary.write("output_size " + str(output_size) + '\n')
    model_summary.write("input_images " + str(len(dataset.input_images)) + '\n')
    model_summary.write("partial_sequences " + str(len(dataset.partial_sequences)) + '\n')
    model_summary.write("next_words " + str(len(dataset.next_words)) + '\n')
   

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    if not is_memory_intensive:
        history = model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        history = model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

    model_history = open(output_path + "model_history" + ".txt", "w+")
    model_history.write(str(history.history.keys()) + "\n")
    model_history.write(str(history.history['acc']) + "\n")
    model_history.write(str(history.history['loss']) + "\n")

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        use_generator = False if len(argv) < 3 else True if int(argv[2]) == 1 else False
        pretrained_weigths = None if len(argv) < 4 else argv[3]

    run(input_path, output_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)
