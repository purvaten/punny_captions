# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle

import numpy as np
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

from inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

from nltk.corpus import stopwords

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("pun_dict", "", "python pickle file containing the pun dictionary.")

tf.logging.set_verbosity(tf.logging.INFO)


cachedStopWords = stopwords.words("english") + ['.']


def remove_stopwords(caption):
  """Return list of tags from caption."""
  tags = [word for word in caption.split() if word not in cachedStopWords]
  return tags


def get_puns(tags):
  """Return a list of puns corresponding to the tags list."""
  with open(FLAGS.pun_dict, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
  puns = [list(p[tags[i]])[j] for i in range(len(tags)) if tags[i] in p for j in range(len(p[tags[i]]))]
  return puns


def get_objs(img_path):
  """ Return list of top 5 object categories predicted by Inception-ResNet-v2 model."""
  model = InceptionResNetV2(include_top=True, weights='imagenet')
  img = image.load_img(img_path, target_size=(299, 299))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds)


def format_objs(object_categories):
  """Return correctly formatted list of object categories."""
  # Applying probability threshold of 0.1 to maintain relevant words only
  obj_strings = [object_categories[0][i][1] for i in range(len(object_categories[0])) if object_categories[0][i][2] > 0.1]
  objs = []
  # For object category with multiple words, split as separate words
  # e.g. chocolate_sauce -> chocolate, sauce
  for string in obj_strings:
    splits = string.split('_')
    objs.extend(splits)
  return objs


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  # Extract the top object categories predicted by Inception-ResNet-v2 model
  img_categories = []
  for filename in filenames:
    object_categories = get_objs(filename)
    categories = format_objs(object_categories)
    img_categories.append(categories)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for index, filename in enumerate(filenames):
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      tags = []
      captions = generator.beam_search(sess, image)
      print("\nCaptions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        words = remove_stopwords(sentence)
        tags = list(set(tags).union(set(words)))

      # Get puns for tags obtained from CaptionGenerator and Inception ResNet model
      tags = list(set(tags).union(set(img_categories[index])))
      puns = get_puns(tags)
      generator = caption_generator.PunnyCaptionGenerator(model, puns, vocab)
      captions = generator.beam_search(sess, image)
      print("\nPunny Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (logp=%f)" % (i, sentence, caption.logprob))


if __name__ == "__main__":
  tf.app.run()
