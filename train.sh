#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=dice &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=dice &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=1 --fold=1 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=dice &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=2 --fold=2 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=dice &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=2 --fold=3 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=dice &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=2 --fold=4 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=dice &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=mix &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=cross_entropy &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=cross_entropy &>> global.log; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=cross_entropy &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=8 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=4 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=mix &>> global.log; \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=4 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=mix &>> global.log \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=4 --epochs=50 --gpu=6 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet --loss_name=cross_entropy &>> global.log; \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=8 --epochs=50 --gpu=6 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=cross_entropy &>> global.log; \
        python ./src/run_training.py --normalize --tumor --discard --batch_size=4 --epochs=50 --gpu=6 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus --loss_name=cross_entropy &>> global.log \
       ) &

