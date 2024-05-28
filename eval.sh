# dafnet "model_name" ZSRE 
python seq_eval.py -en dafnet -mn "model_name" -eds ZSRE -dvc 0 -edvc 1 -sen 1 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds ZSRE -dvc 0 -edvc 1 -sen 10 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds ZSRE -dvc 0 -edvc 1 -sen 100 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds ZSRE -dvc 0 -edvc 1 -sen 1000 -ckpt "ckpt_path"
# dafnet "model_name" CF 
python seq_eval.py -en dafnet -mn "model_name" -eds CF -dvc 0 -edvc 1 -sen 1 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds CF -dvc 0 -edvc 1 -sen 10 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds CF -dvc 0 -edvc 1 -sen 100 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds CF -dvc 0 -edvc 1 -sen 1000 -ckpt "ckpt_path"
# dafnet "model_name" RIPE 
python seq_eval.py -en dafnet -mn "model_name" -eds RIPE -dvc 0 -edvc 1 -sen 1 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds RIPE -dvc 0 -edvc 1 -sen 10 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds RIPE -dvc 0 -edvc 1 -sen 100 -ckpt "ckpt_path"
python seq_eval.py -en dafnet -mn "model_name" -eds RIPE -dvc 0 -edvc 1 -sen 1000 -ckpt "ckpt_path"

# others like ROME
python seq_eval.py -en rome -mn llama -eds CF -dvc 0 -edvc 0 -sen 1 
