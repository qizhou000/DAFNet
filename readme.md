# Before training or testing DAFNet
Please place the Huggingface weight file of the language model to be edited into `models/` directory and write config of DAFNet for this model similar to `configs/dafnet/llama-7b.yaml`. Then add corresponding path into the function `get_model_path_editor_config_name` in `utils/utils.py`. 

# Train DAFNet
Please run:
```
python train_dafnet.py --model_name "model_name" --device 0 --extra_device 1 2 
```
where replace `model_name` with the name of language model you have added into `get_model_path_editor_config_name`. 
Checkpoints will be saved in `train_records/dafnet/model_name/train_name/checkpoints/`.
You can view training information in `train_records/dafnet/model_name/train_name/logs/` through Tensorboard.
# Evaluate DAFNet
In `eval.sh`, please set `-ckpt` as the checkpoint path of DAFNet and set `-mn` as the name of language model to be edited. Then run:
```
sh eval.sh
```
You can check results in `eval_results/dafnet`.

# Extra Editor
If you want to implement a new language model editor, please inherit the base editor class `editors.editor.BaseEditor` and base editor config class `editors.editor.EditorConfig`.

