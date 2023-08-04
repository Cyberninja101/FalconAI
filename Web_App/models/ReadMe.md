cd to /Web_App/models

Run This:

```
python finetune_pytorch.py --model_name_or_path=gpt2 --model_type=gpt2 --train_data_file=/Users/sunzb1/Documents/FalconAI/Web_App/models/test.txt --eval_data_file=/Users/sunzb1/Documents/FalconAI/Web_App/models/train.txt --output_dir=output --do_train=True --do_eval=True
```