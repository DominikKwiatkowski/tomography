#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography


nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --tumor --discard --batch_size=2 --epochs=60 --gpu=0 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --tumor --discard --batch_size=2 --epochs=60 --gpu=0 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=dice &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --multiclass --discard --batch_size=2 --epochs=60 --gpu=1 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --multiclass --discard --batch_size=2 --epochs=60 --gpu=1 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=dice &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --multiclass --discard --batch_size=2 --epochs=60 --gpu=2 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=cross_entropy &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --tumor --discard --batch_size=2 --epochs=60 --gpu=2 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=cross_entropy &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --discard --batch_size=2 --epochs=60 --gpu=3 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=mix &>> global.log; \
        python ./src/run_training.py --ww=225 --wl=25 --optimizer="SGD" --scheduler="Polynomial" --discard --batch_size=2 --epochs=60 --gpu=3 --fold=0 --learning_rate=0.001 --img_size=512 --metadata="/macierz/home/s175454/lits_prepared/metadata.csv" --dataset="/macierz/home/s175454/lits_prepared" --net_name=transformer --loss_name=cross_entropy &>> global.log \
      ) &
