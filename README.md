# Abstract_Goal
Variational modeling of abstract goal for next action anticipation

This code accompines the paper [Predicting the Next Action by Modeling the Abstract Goal](https://arxiv.org/abs/2209.05044)

# Features

  Download RGB, Flow and OBJ features from [RULSTM](https://github.com/fpv-iplab/rulstm) project, specifically this script

  https://github.com/fpv-iplab/rulstm/blob/master/RULSTM/scripts/download_data_ek55.sh


  The data is now in <pwd>/data/ek55/<rgb><flow><obj>. Next, fetch the *training.csv* and *validation.csv* from the RULSTM project [ek55 directory](https://github.com/fpv-iplab/rulstm/tree/master/RULSTM/data/ek55)

# Training
  * We train two models - one for verb and another for noun using each feature RGB/FLOW/OBJ to get 6 models.
  * Example, train a verb model with RGB features
  ``` python main.py  --modality rgb --dataset ek55 --outputs verb --obs_sec 2 --ant_sec 1.0  --latent_dim 128 --num_act_cand 10 --num_goal_cand 3  --hidden_dim 256 --n_layers 1 --nepochs 15 --losses og na ng oa gc --scheduler none --batch_size 256 --sampling 10 ```
  * Explanation of options
    ```--modality', nargs='+', default=None, choices=['rgb', 'flow', 'obj' ], help='Choose tsn (rgb or flow) or obj or vit features or a combination for fusion', required=True)```
parser.add_argument('--dataset', type=str, default='ek55', choices=['ek55', 'ek100', 'egtea'], help='Choose between EK55, EK100 and EGTEA')
parser.add_argument('--outputs', nargs='+', choices=['verb', 'noun', 'action', 'act'], help='Choose between verb and noun or act for EGTEA')
parser.add_argument('--obs_sec',dest='obs_sec', type=int, default=2, choices=[1, 2, 3, 4, 5, 6], help='Choose observed duration in secs 1-6')
parser.add_argument('--ant_sec',dest='ant_sec', type=float, default=1, choices=[2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25], help='Choose anticipation time')
parser.add_argument('--latent_dim', type=int, default=128, help='Choose latent dimension')
parser.add_argument('--num_act_cand', type=int, default=10, help='Choose number of candidates to sample for next verb/noun- 10, 20, 30, 50, 100')
parser.add_argument('--num_goal_cand', type=int, default=1, help='Choose number of abstract goals to sample- 1, 2, 3, 4, 5')
parser.add_argument('--hidden_dim', type=int, default=256, help='Choose hidden dimension')
parser.add_argument('--n_layers', type=int, default=1, choices=[1, 2, 3], help='Choose # layers in RNN for next visual feature - 1, 2, 3')
parser.add_argument('--dropout', type=float, default=0.8, help="Dropout rate")
parser.add_argument('--sampling', type=int, default=6, choices=[1, 2, 3, 5, 6, 10, 15], help='Choose sampling freq of input features, 2, 3, 5, 6, 10, 15')
parser.add_argument('--nepochs', type=int, default=10, help='Choose num epochs')
parser.add_argument('--scheduler', type=str, default='none', choices=['cosine', 'none'], help='Choose scheduler - cosineLR or none(AdamW)')
parser.add_argument('--batch_size', type=int, default=256, choices=[32, 64, 128, 256], help='Choose batch_size - 64, 128, 256')
parser.add_argument('--losses', nargs='+', choices=['og', 'na', 'ng', 'oa', 'gc', 'futmse', 'futce'], help='' )




* Testing on test set
  
  * Fetch the *test_seen.csv* and *test_unseen.csv* from the RULSTM project [ek55 directory](https://github.com/fpv-iplab/rulstm/tree/master/RULSTM/data/ek55)
  
  * The CSV format is different in *training.csv* and *test_seen.csv*. For *training.csv*, the columns are - ```segment_id, video_id, start_frame, end_frame, verb, noun, action``` For *test_seen/unseen.csv*, the columns are - ```segment_id, video_id, start_frame, end_frame```
  
  * We need to train 2 models - one with RGB features as above and another with OBJ features
    Download OBJ features from [RULSTM](https://github.com/fpv-iplab/rulstm) project, specifically this script
    ```
    mkdir -p data/ek55/obj
    curl https://iplab.dmi.unict.it/sharing/rulstm/features/obj/data.mdb -o data/ek55/obj/data.mdb
    ```
    The data is now in <pwd>/data/ek55/obj. 




# Acknowledgment

This research/project is supported in part by the National Research Foundation, Singapore under its AI Singapore Program (AISG Award No: AISG2-RP-2020-016) and the National Research Foundation Singapore under its AI Singapore Program (Award Number: AISG-RP-2019-010).

  
In case of issues, please write to debadityaroy5555 at gmail dot com
