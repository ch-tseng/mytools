import math
import os
import tensorflow as tf
import cv2
import imutils
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

# tell our function where to find the trained model and vocabulary
checkpoint_path = '/media/sf_VMshare/im2txt/Chapter02/model'
vocab_file = '/media/sf_VMshare/im2txt/Chapter02/model/word_counts.txt'
#testfile = '/media/sf_VMshare/im2txt_imgs/1547719853.926031.jpg'
#testfile = '/media/sf_VMshare/im2txt_imgs/1547769267.6346262.jpg'
#testfile = '/media/sf_VMshare/im2txt_imgs/1548411472.3242188.jpg'
testfile = '/media/sf_VMshare/im2txt_imgs/DSC03286.jpg'

def gen_caption(input_files):
    # only print serious log messages
    tf.logging.set_verbosity(tf.logging.FATAL)
    # load our pretrained model
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                 checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(vocab_file)

    filenames = []
    for file_pattern in input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), input_files)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        
        captionlist = []

        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                captionlist.append(sentence)
    return captionlist

cv2.imshow("TEST", imutils.resize(cv2.imread(testfile), width=600))

capts = gen_caption(testfile)
cv2.waitKey(0)
