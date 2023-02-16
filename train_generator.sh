# python3 step_generator.py --preprocessed_data_path='/home/yinhong/Documents/datasets/Kaggle_all_the_news/preprocessed_step_level_data.json'\
#                             --model_saving_path='baseline-checkpoint/step-generator-flanT5/'\
#                             --lr=1e-4\
#                             --l2_decay=0.01\
#                             --epoch=1\
#                             --batch_size=2\
#                             --input_max_length=1024\
#                             --output_max_length=256\
#                             --eval_steps=100000\
#                             --save_steps=200000\
#                             --logging_steps=100\
#                             --warmup_steps=100


python3 paraphrase_generator.py --preprocessed_data_path='/home/yinhong/Documents/datasets/Kaggle_all_the_news/preprocessed_step_level_data.json'\
                            --model_saving_path='baseline-checkpoint/paraphrase-generator-flanT5/'\
                            --lr=1e-4\
                            --l2_decay=0.01\
                            --epoch=1\
<<<<<<< HEAD
                            --batch_size=4\
                            --eval_steps=20000\
                            --save_steps=500000\
                            --logging_steps=500\
=======
                            --batch_size=2\
                            --input_max_length=1024\
                            --output_max_length=1024\
                            --eval_steps=10000\
                            --save_steps=20000\
                            --save_strategy='epoch'\
                            --logging_steps=100\
>>>>>>> a693421612658379d7f80ff666ec96204375ec9b
                            --warmup_steps=100