#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography


nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --multiclass --discard --batch_size=8 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=mix &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --tumor --discard --batch_size=8 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --tumor --discard --batch_size=4 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --tumor --discard --batch_size=4 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=mix &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --discard --batch_size=8 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --discard --batch_size=4 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --discard --batch_size=4 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=mix &>> global.log \
       ) &

