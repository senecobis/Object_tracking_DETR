# Trackformer video Segmentation

Train and test your model with this repo

To do inference, since we didn't diverged particularly from trackformer repo\

Clone commit 958cf89a3d912182e0a67327b493c34232ca8ec4 from the official trackformer repository : https://github.com/timmeinhardt/trackformer
as stated in the following issue : https://github.com/timmeinhardt/trackformer/issues/47

Follow the installation steps in the repo.

Clone this repository, delete the original src and rename this repo as src

Download MOTS20 dataset\

Specify your dataset name or leave EXCAV to use any dataset
Put in /data you folder with images and then ...
Specify data_root_dir to the path to your image folder
Specify obj_detect_checkpoint_file to the path to your model

Before running source and eventually install deformable attention module:
```python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install ```


Then run 

```python new_track.py ```

For Training run 

python src/train.py

  ``` python src/new_train.py ```




    