#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography



nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=4 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=4 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --seed=30 --net_name=transformer --loss_name=mix &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=4 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --seed=31 --net_name=transformer --loss_name=mix &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=4 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --seed=32 --net_name=transformer --loss_name=mix &>> global.log \
      ) &
