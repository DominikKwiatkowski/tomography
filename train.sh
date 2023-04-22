#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography



nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=8 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=24 --n_heads=24 --d_model=1024 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=24 --n_heads=16 --d_model=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=6 --n_heads=4 --d_model=192 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=8 --n_heads=6 --d_model=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=16 --n_heads=16 --d_model=1024 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=20 --n_heads=20 --d_model=1024 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=10 --n_heads=8 --d_model=768 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=2 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=4 --n_heads=2 --d_model=384 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=24 --n_heads=24 --d_model=1024 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="Adam" --multiclass --discard --batch_size=1 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.00001 --img_size=512 --n_layers=24 --n_heads=16 --d_model=1024 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log \
      ) &
