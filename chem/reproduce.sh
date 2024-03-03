mkdir log
for unsup in contextpred infomax edgepred masking graphcl_80 simgrace_80
do
model_file=${unsup}
python finetune.py --input_model_file model_gin/${model_file}.pth --log AdpaterGNN_${model_file}
done
