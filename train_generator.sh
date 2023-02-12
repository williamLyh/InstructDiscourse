python3 step_generator.py --preprocessed_data_path='/home/yinhong/Documents/datasets/Kaggle_all_the_news/preprocessed_step_level_data.json'\
                            --model_saving_path='baseline-checkpoint/step-generator-flanT5/'\
                            --lr=1e-4\
                            --l2_decay=0.01\
                            --epoch=1\
                            --batch_size=4\
                            --eval_steps=20000\
                            --save_steps=500000\
                            --logging_steps=100\
                            --warmup_steps=100