# Punny Captions: Witty Wordplay in Image Descriptions
This is an implementation of the NACL '18 paper "Punny Captions: Witty Wordplay in Image Descriptions", by Arjun Chandrasekaran, Devi Parikh & Mohit Bansal. The forward RNN implementation for new images is in progress. It is complete for images in the [COCO dataset](http://cocodataset.org/).

The pretrained model for the "Show and Tell" [paper](https://arxiv.org/abs/1609.06647) has been used from [here](https://github.com/KranthiGV/Pretrained-Show-and-Tell-model), along with the original paper implementation files in im2txt from [here](https://github.com/tensorflow/models/tree/master/research/im2txt). Changes have been made primarily to im2txt/caption_generator and im2txt/run_inference files.

Full text available at: https://arxiv.org/abs/1704.08224

## Resources
1. JSON file containing pun words associated with coco images : coco_puns.json **_(coming soon)_**
2. Global pun dictionary : pun_dict **_(coming soon)_**
3. Checkpoint file : model.ckpt-2000000 **_(coming soon)_**
4. Coco images folder : http://cocodataset.org/#download
    * Download the 2014 Val images [41K/6GB]

## Steps
0. Follow steps for the Pretrained Show and Tell Model (https://github.com/KranthiGV/Pretrained-Show-and-Tell-model)

1. Add the above punny_captions/im2txt folder in the same directory

2. Set the variables
    * CHECKPOINT_PATH="/path/to/model.ckpt-2000000"
    * VOCAB_FILE="/path/to/word_counts.txt"
    * IMAGE_FILE="/path/to/some-coco-image-file.jpg"
    * COCO_PUNS="/path/to/coco_puns.json"

3. Build the project
    * bazel build -c opt im2txt/run_inference

4. Run the program
    bazel-bin/im2txt/run_inference \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --vocab_file=${VOCAB_FILE} \
    --coco_puns=${COCO_PUNS} \
    --input_files=${IMAGE_FILE}
